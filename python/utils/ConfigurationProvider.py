import os

import configparser


class ConfigurationProvider(object):
    def __init__(self, configuration_file):
        if len(configuration_file) <= 0 or not os.path.exists(configuration_file):
            raise IOError("Failed to find configuration file")
        self.config_parser = configparser.RawConfigParser()
        self.config_parser.read(configuration_file)

    def get_property(self, config_section, config_property, default_value):
        value = str(self.config_parser.get(config_section, config_property))
        if len(value) <= 0:
            return default_value
        return value
