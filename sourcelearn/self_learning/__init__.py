"""Self-Directed Source Learning: M0 (the draft) -> M_S without any task.
No benchmark question is ever seen; M_S is built once per source and shared
by every split.

    state.py    StudyState: steps taken, pending observations, cycles
    planner.py  adaptive study planning (DEEPEN / CONNECT on the observation digest)
    run.py      one cycle: inspection -> adaptive study -> consolidation
"""
