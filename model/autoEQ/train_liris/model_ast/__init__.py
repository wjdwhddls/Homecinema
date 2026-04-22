"""Phase 2a-3 Audio Encoder ablation — AST variant subpackage.

Import submodules explicitly to avoid pytest collection issues with the
parent-package relative imports used downstream in train_liris/dataset.py.
Users should do:

    from model.autoEQ.train_liris.model_ast.config import TrainLirisConfigAST
    from model.autoEQ.train_liris.model_ast.model import AutoEQModelLirisAST
    from model.autoEQ.train_liris.model_ast.dataset import PrecomputedLirisDatasetAST
    from model.autoEQ.train_liris.model_ast.encoders import ASTEncoder
"""
