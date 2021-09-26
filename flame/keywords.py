MODULE_KEY = 'module'
CLASS_KEY = 'class'
KWARGS_KEY = 'kwargs'
EXTRALIBS_KEY = 'extralibs'
NAME_KEY = 'name'
RM_KEY = 'rm_keys'

ENGINE_KEY = 'engine'
HANDLER_KEY = 'handler'
EVENT_KEY = 'event'
HANDLER_KWARGS_KEY = 'handler_kwargs'

EVAL_VALUE_KEY = 'eval_value'

CONFIG_KEY = '_config_'
CHECKPOINTER_KEY = 'checkpointer'

DEFAULT_ENGINE = 'core.engine'
DEFAULT_MODEL = 'core.model'
DEFAULT_BACKUP_CHECKPOINTER = 'handlers.checkpoint.backup'

NOT_EVAL_KEYWORDS = [
    MODULE_KEY,
    CLASS_KEY,
    EXTRALIBS_KEY,
    NAME_KEY,
    RM_KEY,
    HANDLER_KWARGS_KEY,
]
