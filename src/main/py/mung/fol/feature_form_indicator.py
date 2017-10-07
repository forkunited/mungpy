from mung import feature
import dill as pickle

class FeatureFormIndicatorToken(feature.FeatureToken):
    def __init__(self, name, closed_form):
        feature.FeatureToken.__init__(self)
        self._closed_form = closed_form
        self._name = name

    def get_closed_form(self):
        return self._closed_form

    def __str__(self):
        return str(self._closed_form)

    def get_name(self):
        return self._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

class FeatureFormIndicatorType(feature.FeatureType):
    def __init__(self, name, open_form):
        feature.FeatureType.__init__(self)
        self._name = name
        self._open_form = open_form

    def get_name(self):
        return self._name

    def get_open_form(self):
        return self._open_form

    def compute(self, datum, vec, start_index):
        closed_forms = self._open_form.get_closed_forms()
        expr = self._open_form.get_form()
        for i in range(len(closed_forms)):
            c = closed_forms[i]
            if datum.get_model().evaluate(expr, c.get_g())
                vec[start_index + i] = 1.0
            else:
                vec[start_index + i] = 0.0
        return vec

    def get_size(self):
        return len(self._open_form.get_closed_forms())

    def get_token(self, index):
        return FeatureFormIndicatorToken(self._name + "_" + str(index), self._open_form.get_closed_forms()[index])

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeatureFormIndicatorType):
            return False

        my_g = self._open_form.get_init_g()
        g = feature_type.get_open_form().get_init_g()

        for v in my_g:
            if v not in g or g[v] != my_g[v]:
                return False

        return self._open_form.exp_matches(feature_type.get_open_form())

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeatureFormIndicatorType"
        obj["name"] = self._name
        obj["open_form"] = pickle.dumps(self._open_form)
        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            return FeatureFormIndicatorType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        open_form = pickle.loads(obj["open_form"])
        return FeatureFormIndicatorType(name, open_form)

feature.register_feature_type(FeatureFormIndicatorType)
