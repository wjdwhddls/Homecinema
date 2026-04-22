"""Phase 2a-4 Visual Encoder ablation — CLIP frame-mean variant subpackage.

Import submodules explicitly to avoid pytest collection issues with the
parent-package relative imports used downstream in train_liris/dataset.py.
Users should do:

    from model.autoEQ.train_liris.model_clipmean.config import TrainLirisConfigCLIPMean
    from model.autoEQ.train_liris.model_clipmean.model import AutoEQModelLirisCLIPMean
    from model.autoEQ.train_liris.model_clipmean.dataset import PrecomputedLirisDatasetCLIPMean
    from model.autoEQ.train_liris.model_clipmean.encoders import CLIPFrameMeanEncoder
"""
