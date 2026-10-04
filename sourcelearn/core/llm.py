"""Single LLM access point: OpenAI-compatible chat and embeddings (OpenAI,
Azure OpenAI, or OpenRouter for vendor-prefixed model ids such as
"<vendor>/<model>"). All calls are traced with token usage so cost metrics
can be recomputed from trace files.
"""
from __future__ import annotations

import json
import os
import re
import time
import threading
from pathlib import Path
from typing import Any

from sourcelearn.core.trace import NullTrace, Trace

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def load_project_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from a project-local .env file, OVERRIDING any
    inherited environment. Looked up at ./.env unless SOURCELEARN_ENV_FILE
    points elsewhere. A missing file is fine: the inherited env is used as-is."""
    p = Path(path or os.environ.get("SOURCELEARN_ENV_FILE", ".env"))
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")


def _extract_json(text: str) -> Any:
    """Parse a JSON object from raw model text, tolerating code fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    return json.loads(text)


EMBED_CACHE = "data/cache/embeddings"          # append-only store (a symlink to a local disk is fine)
_EMBED_STORES: dict[str, "_EmbedStore"] = {}
_EMBED_STORES_LOCK = threading.Lock()


class _EmbedStore:
    """Append-only on-disk embedding cache shared by every process of every
    run: `keys.txt` (one sha1 per line) + `vectors.f16` (rows x dim float16
    in the same order) + `meta.json` {dim}. Readers take no lock: they memmap
    the vectors and index by `keys.txt`; an append writes the vector bytes
    first and the key lines last (both fsynced), so a key is never visible
    before its row. Writers hold a flock only while appending."""

    SHA = 40

    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        self.keys_p, self.vec_p, self.meta_p = d / "keys.txt", d / "vectors.f16", d / "meta.json"
        self.idx: dict[str, int] = {}
        self._n_bytes = -1
        self._plock = threading.Lock()
        self.dim: int | None = int(json.loads(self.meta_p.read_text())["dim"]) if self.meta_p.exists() else None
        self.refresh()

    def _locked(self):
        import fcntl
        from contextlib import contextmanager

        @contextmanager
        def cm():
            with self._plock:
                lock = open(self.d / ".lock", "w")
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    lock.close()
        return cm()

    def _write(self, shas: list[str], block) -> None:
        import numpy as np
        if self.dim is None:
            self.dim = int(block.shape[1])
            self.meta_p.write_text(json.dumps({"dim": self.dim}))
        with open(self.vec_p, "ab") as f:
            f.write(np.ascontiguousarray(block, dtype=np.float16).tobytes()); f.flush(); os.fsync(f.fileno())
        with open(self.keys_p, "a") as f:
            f.write("".join(s + "\n" for s in shas)); f.flush(); os.fsync(f.fileno())

    def refresh(self) -> None:
        """Re-read the key index when the file grew (another process appended)."""
        if not self.keys_p.exists():
            return
        size = self.keys_p.stat().st_size
        if size == self._n_bytes:
            return
        keys = [k for k in self.keys_p.read_text().split("\n") if len(k) == self.SHA]
        self.idx = {k: i for i, k in enumerate(keys)}
        self._n_bytes = size
        if self.dim is None and self.meta_p.exists():
            self.dim = int(json.loads(self.meta_p.read_text())["dim"])

    def append(self, shas: list[str], block) -> None:
        with self._locked():
            self._n_bytes = -1
            self.refresh()
            keep = [i for i, s in enumerate(shas) if s not in self.idx]   # another process may have added some meanwhile
            if keep:
                self._write([shas[i] for i in keep], block[keep])
            self._n_bytes = -1
            self.refresh()

    def rows(self, rows: list[int]):
        import numpy as np
        n = self.vec_p.stat().st_size // (self.dim * 2)
        mm = np.memmap(self.vec_p, dtype=np.float16, mode="r", shape=(n, self.dim))
        return np.asarray(mm[rows], dtype=np.float32)


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat API."""

    def __init__(self, model: str, provider: str | None = None,
                 trace: Trace | None = None, temperature: float = 0.0,
                 seed: int = 0, reasoning_effort: str | None = None):
        load_project_env()  # project .env overrides inherited credentials
        self.model = model
        self.temperature = temperature
        self.seed = seed
        # per-client reasoning effort (the judge may run at another effort than
        # the backbone); the env variable is the default for every client
        self.reasoning_effort = reasoning_effort or os.environ.get("SOURCELEARN_REASONING_EFFORT") or None
        self.trace = trace or NullTrace()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._usage_lock = threading.Lock()  # parallel batches
        # vendor-prefixed ids ("anthropic/...", "google/...") go through OpenRouter
        if provider is None and "/" in self.model:
            provider = "openrouter"
        self._client = self._make_client(provider)

    @staticmethod
    def _make_client(provider: str | None):
        import openai

        provider = provider or os.environ.get("SOURCELEARN_PROVIDER")
        if provider is None:
            provider = ("openai" if os.environ.get("OPENAI_API_KEY")
                        else "azure" if os.environ.get("AZURE_OPENAI_ENDPOINT")
                        else "openai")
        if provider == "azure":
            return openai.AzureOpenAI()
        if provider == "openrouter":  # OpenAI-compatible; key in .env
            return openai.OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"])
        return openai.OpenAI()

    def embed(self, texts: list[str], model: str = "text-embedding-3-large",
              cache_dir: str | Path = EMBED_CACHE):
        """Embeddings (N x D float32) behind a disk cache keyed by
        (model, sha1(text)); vectors are stored float16. Shared by every
        dense retriever and every process (`_EmbedStore`), so a corpus is
        embedded once."""
        import hashlib

        import numpy as np

        texts = [t if str(t).strip() else "(empty)" for t in texts]   # embedding APIs reject empty input
        d = Path(cache_dir) / model
        with _EMBED_STORES_LOCK:
            store = _EMBED_STORES.get(str(d))
            if store is None:
                store = _EMBED_STORES[str(d)] = _EmbedStore(d)
        store.refresh()
        by_sha = {hashlib.sha1(t.encode()).hexdigest(): t for t in texts}
        missing = sorted(s for s in by_sha if s not in store.idx)
        if missing:
            new = []
            for i in range(0, len(missing), 128):
                batch = [by_sha[s][:8000] for s in missing[i:i + 128]]
                resp = self._embed_create(model, batch)
                new.extend(np.asarray(e.embedding, dtype=np.float16) for e in resp.data)
                with self._usage_lock:
                    self.calls += 1
                    self.prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
                self.trace.event("embed", model=model, n=len(batch), purpose="embed")
            store.append(missing, np.stack(new))
        return store.rows([store.idx[hashlib.sha1(t.encode()).hexdigest()] for t in texts])

    def _embed_create(self, model: str, batch: list[str]):
        """Embeddings come from OpenAI; SOURCELEARN_EMBED_PROVIDER=openrouter
        routes the SAME model through OpenRouter ("openai/<model>", identical
        vectors, so the disk cache keyed by the bare model name stays valid)."""
        import openai

        if not hasattr(self, "_embed_client"):
            if os.environ.get("SOURCELEARN_EMBED_PROVIDER", "openai") == "openrouter":
                self._embed_client = (openai.OpenAI(base_url=OPENROUTER_URL, api_key=os.environ["OPENROUTER_API_KEY"]),
                                      f"openai/{model}")
            else:
                self._embed_client = (openai.OpenAI() if "/" in self.model else self._client, model)
        client, name = self._embed_client
        return client.embeddings.create(model=name, input=batch)

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Reasoning models reject temperature/seed and accept reasoning_effort.
        The raw OpenAI SDK does not drop unsupported params, so we must. The
        o-series / gpt-5 families are built in; other vendors' reasoning models
        are declared with SOURCELEARN_REASONING_MODELS (comma-separated name
        prefixes, matched without the "<vendor>/" part)."""
        extra = os.environ.get("SOURCELEARN_REASONING_MODELS", "")
        prefixes = ("o1", "o3", "o4", "gpt-5", *(p.strip() for p in extra.split(",") if p.strip()))
        return model.split("/")[-1].startswith(prefixes)

    def chat(self, messages: list[dict], purpose: str = "") -> dict:
        """Returns {"content": str|None}."""
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages)
        if self._is_reasoning_model(self.model):
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature
            kwargs["seed"] = self.seed
        resp = self._create_with_choices(kwargs)
        msg = resp.choices[0].message
        usage = resp.usage
        with self._usage_lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += usage.prompt_tokens
                self.completion_tokens += usage.completion_tokens
        self.trace.event(
            "llm_call", purpose=purpose, model=self.model,
            messages=messages, content=msg.content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        return {"content": msg.content}

    EMPTY_REPLY_BACKOFF = (2, 5, 15)     # seconds between attempts

    def _create_with_choices(self, kwargs: dict):
        """Aggregators (OpenRouter) deliver some upstream failures as HTTP 200
        with an error body and no `choices`; the SDK's transport retries never
        see them. A completion has no side effect, so retry, then fail with
        the provider's own error text."""
        for wait in (*self.EMPTY_REPLY_BACKOFF, None):
            resp = self._client.chat.completions.create(**kwargs)
            if getattr(resp, "choices", None):
                return resp
            if wait is None:
                detail = getattr(resp, "error", None) or (getattr(resp, "model_extra", None) or {}).get("error")
                raise RuntimeError(f"{self.model}: completion without choices after {len(self.EMPTY_REPLY_BACKOFF) + 1} attempts: {str(detail)[:300]}")
            time.sleep(wait)

    def complete_json(self, system: str, user: str, schema: dict,
                      purpose: str = "", retries: int = 2) -> dict:
        """Ask for a JSON object matching `schema` (embedded in the prompt for
        broad provider compatibility) and parse it, retrying on parse errors."""
        sys_msg = (
            f"{system}\n\nRespond with a single JSON object matching this JSON "
            f"schema (no prose, no code fences):\n{json.dumps(schema)}"
        )
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            out = self.chat(messages, purpose=purpose or "complete_json")
            try:
                return _extract_json(out["content"] or "")
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                # a runaway / malformed reply must not be replayed in full: the
                # retry prompt would grow past the context window
                messages.append({"role": "assistant", "content": (out["content"] or "")[:2000]})
                messages.append({
                    "role": "user",
                    "content": f"That was not valid JSON ({e}). Reply with only the JSON object.",
                })
        raise ValueError(f"model did not return valid JSON: {last_err}")

    def usage_summary(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}
