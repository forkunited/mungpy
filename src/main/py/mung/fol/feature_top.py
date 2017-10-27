from mung import feature

class FeatureTopToken(feature.FeatureToken):
    def __init__(self, name):
        feature.FeatureToken.__init__(self)
        self._name = name

    def __str__(self):
        return self._name + "_T"

    def get_name(self):
        return self._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass


class FeatureTopType(feature.FeatureType):
    def __init__(self, name):
        feature.FeatureType.__init__(self)
        self._name = name

    def get_name(self):
        return self._name

    def compute(self, datum, vec, start_index):
        vec[start_index] = 1.0
        return vec

    def get_size(self):
        return 1

    def get_token(self, index):
        return FeatureTopToken(self._name)

    def __eq__(self, feature_type):
        return isinstance(feature_type, FeatureTopType) and self._name == feature_type._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeatureTopType"
        obj["name"] = self._name

        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            return FeatureTopType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        return FeatureTopType(name)

feature.register_feature_type(FeatureTopType)

