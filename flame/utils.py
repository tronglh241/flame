import __main__

from importlib import import_module
from torch.utils.data import DataLoader


def create_dataloader(configs):
    dataset = create_instance(configs)
    dataloader = DataLoader(dataset, **configs['dataloader'])
    return dataloader


def create_metrics(metric_module, metric_class, metric_name, metric_params):
    metrics = {}
    for m_module, m_class, m_name, m_params in zip(metric_module, metric_class, metric_name, metric_params):
        for key in m_params:
            if isinstance(m_params[key], str):
                m_params[key] = eval(m_params[key], __main__.__extralibs__)
        metrics[m_name] = getattr(import_module(m_module), m_class)(**m_params)
    return metrics


def create_instance(config):
    module = config['module']
    class_ = config['class']
    config_kwargs = config.get(class_, {})
    for key, value in config_kwargs.items():
        if isinstance(value, str):
            config_kwargs[key] = eval(value, __main__.__extralibs__)
    return getattr(import_module(module), class_)(**config_kwargs)


def load_yaml(yaml_file):
    import yaml
    with open(yaml_file) as f:
        settings = yaml.safe_load(f)
    return settings
