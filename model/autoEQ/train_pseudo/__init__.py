"""CogniMuse-only training pipeline.

See PLAN.md for full design rationale. Mirrors the architecture of
`model/autoEQ/train/` minus the Congruence head / negative sampling,
plus LOMO cross-validation + time-based within-movie val holdout.
"""
