from importlib import import_module
from typing import Any

from fvcore.common.config import CfgNode as _CfgNode

from ...keywords import (CLASS_KEY, EXTRALIBS_KEY, MODULE_KEY, NAME_KEY,
                         NOT_EVAL_KEYWORDS, RM_KEY)


class CfgNode(_CfgNode):
    def eval(self) -> Any:
        def _eval(config: Any) -> Any:
            if isinstance(config, dict):
                for key, value in config.items():
                    if key not in NOT_EVAL_KEYWORDS:
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

        # Generate extra libs
        for alias, lib_info in config.pop(EXTRALIBS_KEY, {}).items():
            if isinstance(lib_info, dict):
                module = lib_info[MODULE_KEY]
                name = lib_info[NAME_KEY]
                lib = getattr(import_module(module), name)
            else:
                lib = import_module(lib_info)

            extralibs[alias] = lib

        # Eval config
        config = _eval(config)

        # Remove unnecessary keys
        for rm_key in config.pop(RM_KEY, []):
            del config[rm_key]

        return config

    def __delitem__(self, name: str) -> None:
        name_parts = name.split('.')
        dic = self

        for name_part in name_parts[:-1]:
            dic = dic[name_part]

        super(CfgNode, dic).__delitem__(name_parts[-1])

    def get(self, name: str, default: Any = None) -> None:
        name_parts = name.split('.')
        dic = self

        for name_part in name_parts[:-1]:
            dic = dic[name_part]

        return super(CfgNode, dic).get(name_parts[-1], default)


global_cfg = CfgNode(new_allowed=True)
