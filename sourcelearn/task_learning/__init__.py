"""Task learning on a source model: two direct forms of task supervision.

    Failure-Guided Local Refinement      a task that fails exposes a source
        region the current M did not support well enough; the region is
        re-read and refined so that what the task required from the source is
        represented explicitly (refine.refine_region).
    Task-Induced Representation Calibration    every task, solved or not,
        shows what the source is used for; each yields a representation
        lesson (lesson.policy_lesson), recurring lessons consolidate into
        the representation policy Pi (aggregate.aggregate), and the policy is
        applied back to the whole model (recalibrate.recalibrate).

    wrong tasks improve where M is weak; all tasks teach M what tends to matter.

evidence.py holds what a task required from the source (E*) versus what the
model offered (E^M); splits.py the seeded train/test split; run.py drives
one training pass -> one policy -> the derived models -> repeated evaluation.
"""
