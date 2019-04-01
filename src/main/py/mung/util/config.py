import json

class Config:
    def __init__(self, properties, environment=None):
        self._properties = properties
        self._environment = environment
        if environment is not None:
            self._set_properties_environment(properties, environment)

    def __getitem__(self, key):
        return self._properties[key]

    def __contains__(self, key):
        return (key in self._properties)

    def __iter__(self):
        return iter(self._properties)

    def get_dict(self):
        return self._properties

    def get_environment(self):
        return self._environment

    def _set_properties_environment(self, properties, environment):
        if isinstance(properties, list):
            for i in range(len(properties)):
                if isinstance(properties[i], str):
                    properties[i] = self._set_str_environment(properties[i], environment)
                else:
                    self._set_properties_environment(properties[i], environment)
        elif isinstance(properties, dict):
            for key in properties.keys():
                item = properties[key]
                if isinstance(item, str):
                    properties[key] = self._set_str_environment(item, environment)
                else:
                    self._set_properties_environment(item, environment)

    def _set_str_environment(self, s, environment):
        for key in environment:
            s = s.replace("$!{" + key + "}", environment[key])
        return s

    def merge(self, other_config):
        environment = None
        if self._environment is not None or other_config._environment is not None:
            environment = dict()
            if self._environment is not None:
                for key in self._environment:
                    environment[key] = self._environment[key]
            if other_config._environment is not None:
                for key in other_config._environment:
                    environment[key] = other_config._environment[key]
        
        properties = dict()
        for key in self._properties:
            properties[key] = self._properties[key]
        for key in other_config._properties:
            properties[key] = other_config._properties[key]

        return Config(properties, environment=environment)

    @staticmethod
    def load(properties_file, environment=None):
        properties = None
        with open(properties_file, 'r') as fp:
            properties = json.load(fp)

        return Config(properties, environment=environment)

    @staticmethod
    def load_from_list(args_list, environment=None):
        properties = dict()

        if len(args_list) % 2 != 0:
            raise ValueError("Args list must be even length to parse into config")

        for i in range(int(len(args_list)/2)):
            arg_name = args_list[i*2].replace("--", "").replace("-", "")
            arg_value = args_list[i*2+1]

            properties[arg_name] = arg_value

        return Config(properties, environment=environment)

    @staticmethod
    def load_from_dict(d, environment=None):
        properties = dict(d)
        return Config(properties, environment=environment)
