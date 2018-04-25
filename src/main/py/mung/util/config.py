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
                if isinstance(properties[i], basestring):
                    properties[i] = self._set_str_environment(properties[i], environment)
                else:
                    self._set_properties_environment(properties[i], environment)
        elif isinstance(properties, dict):
            for key in properties.keys():
                item = properties[key]
                if isinstance(item, basestring):
                    properties[key] = self._set_str_environment(item, environment)
                else:
                    self._set_properties_environment(item, environment)

    def _set_str_environment(self, s, environment):
        for key in environment:
            s = s.replace("$!{" + key + "}", environment[key])
        return s

    @staticmethod
    def load(properties_file, environment=None):
        properties = None
        with open(properties_file, 'r') as fp:
            properties = json.load(fp)

        return Config(properties, environment=environment)
