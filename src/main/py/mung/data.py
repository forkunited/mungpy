import numpy as np
import copy
from jsonpath_ng import jsonpath, parse
from os import listdir
from os.path import isfile, join
import json

class Datum:
    def __init__(self, properties=dict())
        self._properties = properties

        if "id" not in properties or "type" not in properties:
            raise ValueError("Datum properties object must have an id and a type")

    def get_id(self):
        return self._properties["id"]

    def get_type(self):
        return self._properties["type"]

    def has(self, path):
        return len([match.value for match in parse(path).find(self._properties)]) > 0

    def get(self, path, first=True, include_paths=False):
        # FIXME: This should handle reference datums 
        # at some point
        path_values = [(match.full_path, match.value) for match in parse(path).find(self._properties)]
        if first:
            if len(values) == 0:
                return None
            else:
                if include_paths:
                    return path_values[0]
                else:
                    return path_values[0][1]
        else:
            if include_paths:
                return path_values
            else:
                return [path_value[1] for path_value in path_values]
 
    def get_mutable(self):
        return MutableDatum(properties=copy.deepcopy(self._properties))

    def __eq__(self, other):
        if isinstance(other, Datum):
            return self._properties["id"] == other._properties["id"]
        return NotImplemented

    def __hash__(self):
        return hash(self._properties["id"])

    def save(self, dir_path, name=None):
        if name is None:
            name = self.get_id()
        with open(join(dir_path, name), 'w') as fp:
            json.dump(self._properties, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            return Datum(properties=json.load(fp))


class MutableDatum(Datum):
    def __init__(self, properties=dict()):
        Datum.__init__(self, properties)

    def set(self, key, value, path=None):
        objs = [self._properties]
        if path is not None:
            objs = self.get(path, first=False)
        for obj in objs:
            obj[key] = value


class DatumReference():
    def __init__(self, datum, path):
        self._datum = datum
        self._path = path

    def get_datum(self):
        return self._datum

    def get_path(self):
        return self._path

    def get(self):
        return self._datum.get(self._path)


class DataSet:
    def __init__(self, data=[]):
        self._data = list(data)

    def get(self, i):
        return self._data[i]

    def get_data(self):
        return self._data

    def get_size(self):
        return len(self._data)

    def copy(self):
        return DataSet(data=list(self._data))

    def shuffle(self):
        perm = np.random.permutation(len(self._data))
        shuffled_data = []
        for i in range(len(perm)):
            shuffled_data.append(self._data.get(perm[i]))
        self._data = shuffled_data

    def split(self, sizes):
        datas = []
        index = 0
        for size in sizes:
            part_data = []
            max_index = min(len(self._data), index + size*len(self._data))
            while index < max_index:
                part_data.append(self._data[index])
                index += 1
            datas.append(DataSet(data=part_data))
        return datas

    def save(self, data_dir):
        for datum in self._data:
            datum.save(data_dir)

    @staticmethod
    def load(data_dir):
        D = DataSet()
        files = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        for f in files:
            D._data.append(Datum.load(f))
        D.shuffle() # Ensure deterministic order if random seeded
        return D
