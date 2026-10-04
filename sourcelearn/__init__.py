"""SourceLearn: developing source-specific competence over a persistent authoritative source.

The package maintains an explicit, revisable source model M over a source D and
develops it in three steps, every persistent update being reconstructed from the
source itself (see README and `sourcelearn.pipeline`):

    M_0 = Construct(D; Pi_0)          source model construction        draft/
    M_S = SelfLearn(D, M_0)           Self-Directed Source Learning    self_learning/
    M_T = TaskLearn(D, M_S; G)        Task-Guided Source Learning      task_learning/

Entry points:
    python -m sourcelearn.pipeline    the whole method on one source
    python -m sourcelearn.datasets    the five benchmarks of the paper as pipeline inputs
"""

__version__ = "0.1.0"
