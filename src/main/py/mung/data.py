import numpy as np
import copy
import abc
import os
from jsonpath_ng import jsonpath
from jsonpath_ng.ext import parse
from os import listdir
from os.path import isfile, join
import json

JSON_PATH_CACHE = dict()

class Datum:
    def __init__(self, properties=dict(), id_key="id", path_map=None):
        self._properties = properties
        self._id_key = id_key
        self._path_map = path_map

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
        if self._path_map is not None and path in self._path_map and self.has(self._path_map[path]):
            return True
        return len([match.value for match in Datum.parse_path(path).find(self._properties)]) > 0

    def get(self, path, first=True, include_paths=False):
        # FIXME: This should handle reference datums
        # at some point
        if self._path_map is not None and path in self._path_map:
            return self.get(self._path_map[path], first=first, include_paths=include_paths)

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

    def to_dict(self):
        return self._properties

    def save(self, dir_path, name=None):
        if name is None:
            name = self.get_id()
        with open(join(dir_path, name), 'w') as fp:
            json.dump(self._properties, fp)

    @classmethod
    def load(cls, file_path, id_key="id"):
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
        if path is not None and path != ".":
            objs = self.get(path, first=False)
        if objs is not None and len(objs) > 0:
            for obj in objs:
                obj[key] = value
        else:
            path_keys = path.split(".")
            cur_obj = self._properties
            for path_key in path_keys:
                if path_key not in cur_obj:
                    cur_obj[path_key] = dict()
                cur_obj = cur_obj[path_key]
            cur_obj[key] = value


class DatumReference:
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
    def __init__(self, data=[], id_key="id", source_dir=None):
        self._data = list(data)
        self._id_key = id_key
        self._source_dir = source_dir

    def get_id_key(self):
        return self._id_key

    def get_directory(self):
        return self._source_dir

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return self._data.__iter__()

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
        return perm

    def sort(self, key_fn):
        perm = range(len(self._data))
        indexed_key_fn = lambda index : key_fn(self._data[index])
        perm.sort(key=indexed_key_fn)
        sorted_data = []
        for i in range(len(perm)):
            sorted_data.append(self._data[perm[i]])
        self._data = sorted_data
        return np.array(perm)

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

    def filter(self, filter_fn):
        data = []
        for i in range(self.get_size()):
            if filter_fn(self.get(i)):
                data.append(self.get(i))
        return DataSet(data=data)

    def partition(self, partition, key_fn):
        return partition.split(self, key_fn)

    def union(self, other):
        data_union = []
        data_union.extend(self._data)
        data_union.extend(other._data)
        return DataSet(data=data_union, id_key=self._id_key, source_dir=None)

    def histogram(self, field):
        hist = dict()
        for d in self._data:
            vals = d.get(field)
            if not isinstance(vals, list):
                vals = [vals]
            for val in vals:
                if val is None:
                    continue
                elif isinstance(val, unicode) or isinstance(val, str) or \
                    isinstance(val, float) or isinstance(val, int):
                    if val not in hist:
                        hist[val] = 0
                    hist[val] += 1
                else:
                    raise ValueError("Unsupported histogram value: " + str(val))
        return hist

    def save(self, data_dir, batch=1000):
        self._source_dir = data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if batch is None:
            for datum in self._data:
                datum.save(data_dir)
        else:
            cur_batch = []
            i = 0
            for datum in self._data:
                cur_batch.append(datum.to_dict())
                if len(cur_batch) == batch:
                    with open(join(data_dir, str(i)), 'w') as fp:
                        for datum_dict in cur_batch:
                            fp.write(json.dumps(datum_dict) + "\n")
                    i += 1
                    cur_batch = []

            if len(cur_batch) > 0:
                with open(join(data_dir, str(i)), 'w') as fp:
                    for datum_dict in cur_batch:
                        fp.write(json.dumps(datum_dict) + "\n")

    @classmethod
    def load(cls, data_dir, id_key="id", order=None, path_map=None, limit=None):
        D = DataSet(id_key=id_key, source_dir=data_dir)
        file_names = sorted([int(f) for f in listdir(data_dir)])
        files = [join(data_dir, str(f)) for f in file_names if isfile(join(data_dir, str(f)))]

        for f in files:
            with open(f, 'r') as fp:
                json_strs = fp.readlines()
                reached_limit = False
                for json_str in json_strs:
                    D._data.append(Datum(properties=json.loads(json_str.strip()),
                                         id_key=id_key, path_map=path_map))
                    if limit is not None and limit > 0 and len(D._data) >= limit:
                        reached_limit = True
                        break
                if reached_limit:
                    break

        if order is None:
            D.shuffle() # Ensure deterministic order if random seeded
        else:
            if len(D._data) != len(order):
                raise ValueError("Size of ordering for data must match data size")
            keys_to_indices = dict()
            for i in range(len(D._data)):
                keys_to_indices[D._data[i].get_id()] = i
            ordered_data = [D._data[keys_to_indices[order[i]]] for i in range(len(order))]
            D._data = ordered_data

        print "Loaded data of size " + str(D.get_size()) + "."
        return D

class Partition:
    def __init__(self):
        self._keep_data = False
        self._size = 0
        self._parts = dict()

    def copy(self):
        P = Partition()
        P._keep_data = self._keep_data
        P._size = self._size
        P._parts = dict(self._parts)
        return P

    def get_size(self):
        return self._size

    def has_data(self):
        return self._keep_data

    def split(self, data, key_fn):
        split_data = dict()
        for i in range(data.get_size()):
            key = key_fn(data.get(i))
            for part_name in self._parts:
                if key in self._parts[part_name]:
                    if part_name not in split_data:
                        split_data[part_name] = []
                    split_data[part_name].append(data.get(i))

        for key, data in split_data.iteritems():
            split_data[key] = DataSet(data=data)

        return split_data

    def get_part_names(self):
        return self._parts.keys()

    def get_part(self, name):
        return self._parts[name]

    def part_contains(self, name, value):
        return name in self._parts and value in self._parts[name]

    def remove_parts(self, part_names):
        for part_name in part_names:
            self._size -= len(self._parts[part_name])
            del self._parts[part_name]
        return self

    def merge_parts(self, part_names, new_part_name):
        new_part_dict = dict()
        for part_name in part_names:
            new_part_dict.update(self._parts[part_name])
            del self._parts[part_name]
        self._parts[new_part_name] = new_part_dict
        return self

    def split_part(self, part_name, new_part_names, new_sizes):
        old_part_dict = self._parts[part_name]
        del self._parts[part_name]

        old_part_keys = []
        old_part_keys.extend(old_part_dict.keys())
        old_part_values = []
        old_part_values.extend(old_part_dict.values())
        
        cur_start = 0
        for i in range(len(new_part_names)):
            new_part_name = new_part_names[i]
            new_size = int(new_sizes[i]*self._size)

            if i == len(new_part_names) - 1:
                new_size = len(old_part_keys) - cur_start
            
            self._parts[new_part_name] = dict()
            for j in range(cur_start, cur_start+new_size):
                self._parts[new_part_name][old_part_keys[j]] = old_part_values[j]

            cur_start += new_size
        return self

    def union(self, other_part):
        if self._keep_data != other_part._keep_data:
            raise ValueError("Cannot merge a data-included partition with a data-removed partition.")
        P = Partition()
        P._keep_data = self._keep_data
        P._size = self._size + other_part._size
        for key in self._parts.keys():
            P._parts[key] = self._parts[key].copy()
        for key in other_part._parts.keys():
            if key not in P._parts:
                P._parts[key] = dict()
            for item in other_part._parts[key].keys():
                for key2 in P._parts.keys():
                    if item in P._parts[key2]:
                        raise ValueError("Duplicate item in partitions (" + item + ")")
                P._parts[key][item] = other_part._parts[key][item]
        return P

    def save(self, file_path):
        with open(file_path, 'w') as fp:
            obj = dict()
            obj["keep_data"] = self._keep_data
            obj["size"] = self._size
            obj["parts"] = self._parts
            json.dump(obj, fp)

    def make_reverse_lookup(self):
        reverse_lookup = dict()
        for part_key, part_dict in self._parts.iteritems():
            for key in part_dict.keys():
                reverse_lookup[key] = part_key
        return reverse_lookup

    @classmethod
    def load(cls, file_path):
        P = Partition()
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            if "keep_data" in obj:
                P._keep_data = obj["keep_data"]
            else:
                P._keep_data = False
            P._size = obj["size"]
            P._parts = obj["parts"]
        return P

    @classmethod
    def make(cls, D, sizes, part_names, key_fn, maintain_partition=None):
        P = Partition()
        D_parts = D.split(sizes)

        P._keep_data = False
        P._size =  D.get_size()
        P._parts = dict()

        maintain_lookup = None
        if maintain_partition is not None:
            maintain_lookup = dict()
            for part_name in maintain_partition._parts.keys():
                for key in maintain_partition._parts[part_name].keys():
                    maintain_lookup[key] = part_name
                P._parts[part_name] = dict()

        for i in range(len(part_names)):
            if part_names[i] not in P._parts:
                P._parts[part_names[i]] = dict()
            for datum in D_parts[i]:
                key = key_fn(datum)
                if maintain_lookup is not None and key in maintain_lookup:
                    P._parts[maintain_lookup[key]][key] = 1
                else:
                    P._parts[part_names[i]][key] = 1
        return P
