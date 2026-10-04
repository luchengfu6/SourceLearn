"""Answering with a source model and scoring the answer.

`answer` holds Retrieve_M (MRetriever), one task attempt and the comparison
with the reference; `evaluate` is the evaluation harness (hybrid raw
excerpts, with or without Retrieve_M); `feedback` turns a questions.json row
into the task-feedback packet F read by task-guided learning.
"""
