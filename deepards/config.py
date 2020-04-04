import yaml


class Configuration(object):
    def __init__(self, parser_args, config_file):
        with open(config_file) as yml:
            conf = yaml.load(yml)
