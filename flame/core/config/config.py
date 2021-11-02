from importlib import import_module
from typing import Any, Tuple

from yacs.config import CfgNode as _CfgNode

from flame.keywords import (CLASS_KEY, EVAL_VALUE_KEY, EXTRALIBS_KEY,
                            KWARGS_KEY, MODULE_KEY, NAME_KEY,
                            NOT_EVAL_KEYWORDS, RM_KEY)


class CfgNode(_CfgNode):
    @staticmethod
    def _eval(config: Any, global_context: dict, local_context: dict, eval_all: bool = False) -> Any:
        if isinstance(config, dict):
            for key, value in config.items():
                if eval_all or key not in NOT_EVAL_KEYWORDS:
                    config[key] = CfgNode._eval(value, global_context, local_context)

            if MODULE_KEY in config and CLASS_KEY in config:
                module = config[MODULE_KEY]
                class_ = config[CLASS_KEY]
                config_kwargs = config.get(KWARGS_KEY, {})
                return getattr(import_module(module), class_)(**config_kwargs)

        elif isinstance(config, list):
            config = list(map(lambda ele: CfgNode._eval(ele, global_context, local_context), config))

        elif isinstance(config, str):
            config = eval(config, {**global_context, **local_context})

        return config

    def eval(self) -> Tuple[Any, dict]:
        # If config was evaluated, return evaluated value
        if self.__dict__.get(EVAL_VALUE_KEY) is not None:
            return self.__dict__[EVAL_VALUE_KEY]

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

        # Save config
        self.__dict__[EVAL_VALUE_KEY] = config, extralibs

        # Eval config
        config = CfgNode._eval(config, extralibs, org_config)

        # Remove unnecessary keys
        if isinstance(config, dict):
            for rm_key in config.pop(RM_KEY, []):
                del config[rm_key]

        self.freeze()

        return config, extralibs

    def eval_key(self, key: str) -> Any:
        config, extralibs = self.eval()

        try:
            value = eval(key, {}, {**config, **extralibs})
        except Exception:
            value = None

        return value

    def __delitem__(self, name: str) -> None:
        name_parts = name.split('.')
        dic = self

        for name_part in name_parts[:-1]:
            dic = dic[name_part]

        super(CfgNode, dic).__delitem__(name_parts[-1])

    def get(self, name: str, default: Any = None) -> Any:
        name_parts = name.split('.')
        dic = self

        for name_part in name_parts[:-1]:
            dic = dic.get(name_part, CfgNode())

        return super(CfgNode, dic).get(name_parts[-1], default)

    def state_dict(self) -> str:
        return self.dump(sort_keys=False)


global_cfg = CfgNode(new_allowed=True)
