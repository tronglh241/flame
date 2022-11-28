from yacs.config import CfgNode

Keyword = CfgNode()
Keyword.NOT_EVAL = CfgNode()
Keyword.NOT_EVAL.MODULE = 'module'
Keyword.NOT_EVAL.NAME = 'name'
Keyword.NOT_EVAL.AKWARGS = 'akwargs'
Keyword.NOT_EVAL.FUNCTION = 'function'
Keyword.NOT_EVAL.CONFIG = '_config_'
Keyword.MODULE = Keyword.NOT_EVAL.MODULE
Keyword.NAME = Keyword.NOT_EVAL.NAME
Keyword.AKWARGS = Keyword.NOT_EVAL.AKWARGS
Keyword.FUNCTION = Keyword.NOT_EVAL.FUNCTION
Keyword.CONFIG = Keyword.NOT_EVAL.CONFIG
Keyword.KWARGS = 'kwargs'
Keyword.EXTRALIBS = 'extralibs'
Keyword.EVAL_VALUE = 'eval_value'
Keyword.HANDLER = 'handler'
Keyword.EVENT = 'event'
Keyword.ENGINE = 'engine'
Keyword.CONTEXT = 'context'
Keyword.CHECKPOINTER = 'checkpointer'
Keyword.RANK = 'rank'
Keyword.BARRIER = 'barrier'
