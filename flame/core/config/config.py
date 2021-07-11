from importlib import import_module
from typing import Any

from fvcore.common.config import CfgNode as _CfgNode

MODULE_KEY = 'module'
CLASS_KEY = 'class'
EXTRALIBS_KEY = 'extralibs'
NAME_KEY = 'name'
RM_KEY = 'rm_keys'

KEYWORDS = [
    MODULE_KEY,
    CLASS_KEY,
    EXTRALIBS_KEY,
    NAME_KEY,
    RM_KEY,
]


class CfgNode(_CfgNode):
    def eval(self) -> Any:
        def _eval(config: Any) -> Any:
            if isinstance(config, dict):
                for key, value in config.items():
                    if key not in KEYWORDS:
                        config[key] = _eval(value)

                if MODULE_KEY in config and CLASS_KEY in config:
                    module = config[MODULE_KEY]
                    class_ = config[CLASS_KEY]
                    config_kwargs = config.get(class_, {})
                    return getattr(import_module(module), class_)(**config_kwargs)

            elif isinstance(config, list):
                config = list(map(lambda ele: _eval(ele), config))

            elif isinstance(config, tuple):
                config = tuple(map(lambda ele: _eval(ele), config))

            elif isinstance(config, str):
                config = eval(config, extralibs, org_config)

                if not isinstance(config, str):
                    config = _eval(config)

            return config

        config = org_config = self.clone()
        extralibs = {}

        for alias, lib_info in config.pop(EXTRALIBS_KEY, {}).items():
            if isinstance(lib_info, dict):
                module = lib_info[MODULE_KEY]
                name = lib_info[NAME_KEY]
                lib = getattr(import_module(module), name)
            else:
                lib = import_module(lib_info)

            extralibs[alias] = lib

        config = _eval(config)

        for rm_key in config.pop(RM_KEY, []):
            del config[rm_key]

        return config

    def __delitem__(self, name):
        name_parts = name.split('.')
        dic = self

        for name_part in name_parts[:-1]:
            dic = dic[name_part]

        del dic[name_parts[-1]]
