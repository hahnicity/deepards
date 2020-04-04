import os

import yaml


class Configuration(object):
    def __init__(self, parser_args):
        with open(os.path.join(os.path.dirname(__file__), 'defaults.yml')) as defaults:
            self.conf = yaml.load(defaults)

        if parser_args.config_override:
            with open(override_config_file) as overrides:
                for k, v in overrides.items():
                    self.conf[k] = v

        for k, v in parser_args.__dict__.items():
            if v is not None or k not in self.conf:
                self.conf[k] = v

    def __getattr__(self, attr):
        return self.conf[attr]
