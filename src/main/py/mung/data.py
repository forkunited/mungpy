import numpy as np
import copy
import abc
from jsonpath_ng import jsonpath
from jsonpath_ng.ext import parse
from os import listdir
from os.path import isfile, join
import json

JSON_PATH_CACHE = dict()

class Datum:
    def __init__(self, properties=dict(), id_key="id"):
        self._properties = properties
        self._id_key = id_key

        if id_key not in properties:
            raise ValueError("Datum properties object must have an id")

    def get_id(self):
        return self._properties[self._id_key]

    def get_type(self):
        if "type" not in self._properties:
            return None
        else:
            return self._properties["type"]

    def has(self, path):
        return len([match.value for match in Datum.parse_path(path).find(self._properties)]) > 0

    def get(self, path, first=True, include_paths=False):
        # FIXME: This should handle reference datums 
        # at some point
        path_values = [(str(match.full_path), match.value) for match in Datum.parse_path(path).find(self._properties)]
        if first:
            if len(path_values) == 0:
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
        return MutableDatum(properties=copy.deepcopy(self._properties), id_key=self._id_key)

    def __eq__(self, other):
        if isinstance(other, Datum):
            return self._properties[self._id_key] == other._properties[self._id_key]
        return NotImplemented

    def __hash__(self):
        return hash(self._properties[self._id_key])

    def save(self, dir_path, name=None):
        if name is None:
            name = self.get_id()
        with open(join(dir_path, name), 'w') as fp:
            json.dump(self._properties, fp)

    @staticmethod
    def load(file_path, id_key="id"):
        with open(file_path, 'r') as fp:
            properties = json.load(fp)
            return Datum(properties=properties, id_key=id_key)

    @staticmethod
    def parse_path(path_json, cache=True):
        if not cache:
            return parse(path_json)
        if path_json not in JSON_PATH_CACHE:
            JSON_PATH_CACHE[path_json] = parse(path_json)        
        return JSON_PATH_CACHE[path_json]


class MutableDatum(Datum):
    def __init__(self, properties=dict(), id_key="id"):
        Datum.__init__(self, properties, id_key=id_key)

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

    def get_type(self):
        obj = self.get()
        if "type" in obj:
            return obj["type"]
        else:
            return None

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
            shuffled_data.append(self._data[perm[i]])
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
    def load(data_dir, id_key="id"):
        D = DataSet()
        files = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
        for f in files:
            D._data.append(Datum.load(f, id_key=id_key))
        D.shuffle() # Ensure deterministic order if random seeded
        return D
