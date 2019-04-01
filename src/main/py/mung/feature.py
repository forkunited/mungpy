
import numpy as np
import abc
import json
import dill as pickle
import os.path
import torch
from torch.autograd import Variable
from mung import data
from mung.data import DataSet, Partition
from mung.util.rep import StoredVectorDictionary
from bidict import bidict
from collections import Counter
from os.path import join, isfile
from os import listdir
from functools import cmp_to_key


pickle.settings['recurse'] = True

FEATURE_TYPES = dict()
FEATURE_SEQ_TYPES = dict()

def register_feature_type(feature_type):
    FEATURE_TYPES[feature_type.__name__] = feature_type

def register_feature_seq_type(feature_seq_type):
    FEATURE_SEQ_TYPES[feature_seq_type.__name__] = feature_seq_type

class SubsetType:
    RANDOM = "RANDOM"
    FIRST = "FIRST"
    FILTER = "FILTER"
    PARTITION = "PARTITION"

class ValueType:
    ENUMERABLE_ONE_HOT = 0
    ENUMERABLE_INDEX = 1
    SCALAR = 2

class Symbol:
    SEQ_START = "#start#"
    SEQ_MID = "#mid#"
    SEQ_END = "#end#"
    SEQ_UNC = "#unc#"

    @staticmethod
    def index(symbol):
        if symbol == Symbol.SEQ_UNC:
            return 0
        elif symbol == Symbol.SEQ_START:
            return 1
        elif symbol == Symbol.SEQ_END:
            return 2
        elif symbol == Symbol.SEQ_MID:
            return 3
        else:
            return None

class ArrayFormat:
    NUMPY = 0
    TORCH = 1

    @staticmethod
    def cast(arr, form, ints=False):
        if form == ArrayFormat.TORCH:
            t = torch.from_numpy(arr)
            if not ints:
                return t.float()
            else:
                return t.long()
        else:
            return arr

class FeatureToken(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        """ Returns a string representation of the feature """

    @abc.abstractmethod
    def get_name(self):
        """ Returns the name of the feature token """

    @abc.abstractmethod
    def init_start(self):
        """ Start initializing the feature """

    @abc.abstractmethod
    def init_datum(self, datum):
        """ Perform initialization for a given datum """

    @abc.abstractmethod
    def init_end(self):
        """ End initializing the feature """


class FeatureType(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def get_token_count(self):
        return self.get_size()

    def save(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.to_dict(), fp)

    @abc.abstractmethod
    def get_name(self):
        """ Returns the name of the feature type """

    @abc.abstractmethod
    def get_size(self):
        """ Returns the size of the vectors computed by this feature """

    @abc.abstractmethod
    def compute(self, datum, vec, start_index):
        """ Fills in vector vec with feature values starting at start_index """

    @abc.abstractmethod
    def get_token(self, index):
        """ Returns a token of this feature type """

    @abc.abstractmethod
    def __eq__(self, feature_type):
        """ Determines whether two feature types are the same """

    @abc.abstractmethod
    def init_start(self):
        """ Start initializing the feature """

    @abc.abstractmethod
    def init_datum(self, datum):
        """ Perform initialization for a given datum """

    @abc.abstractmethod
    def init_end(self):
        """ End initializing the feature """

    @abc.abstractmethod
    def to_dict(self):
        """ Gives a dictionary representing the feature type """


class FeatureSequence(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, feature_seq):
        """ Returns whether two feature sequences are the same """

    @abc.abstractmethod
    def get_name(self):
        """ Returns the name of the feature sequence """

    @abc.abstractmethod
    def get_size(self):
        """ Returns the padded length of the feature sequence """

    @abc.abstractmethod
    def get_datum_length(self, datum):
        """ Returns the unpadded length of the datum feature sequence """

    @abc.abstractmethod
    def get_type(self, index):
        """ Returns a feature type from the sequence """

    @abc.abstractmethod
    def init_start(self):
        """ Start initializing the feature sequence """

    @abc.abstractmethod
    def init_datum(self, datum):
        """ Perform initialization for a given datum """

    @abc.abstractmethod
    def init_end(self):
        """ End initializing the feature sequence """

    @abc.abstractmethod
    def save(self, file_path):
        """ Save a representation of the feature sequence to file """


class FeatureMatrixToken(FeatureToken):
    def __init__(self, name, index):
        FeatureToken.__init__(self)
        self._name = name
        self._index = index

    def __str__(self):
        return self._name + "_" + str(self._index)

    def get_name(self):
        return self._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

class FeatureMatrixType(FeatureType):
    def __init__(self, name, matrix_fn, size, index=None):
        FeatureType.__init__(self)
        self._name = name
        self._matrix_fn = matrix_fn
        self._size = size
        self._index = index

    def compute(self, datum, vec, start_index):
        if self._index is None:
            vec[start_index:start_index+self._size] = self._matrix_fn(datum).flatten()
        else:
            vec[start_index:start_index+self._size] = self._matrix_fn(datum, self._index).flatten()
        return vec

    def get_name(self):
        return self._name

    def get_size(self):
        return self._size

    def get_token(self, index):
        return FeatureMatrixToken(self._name, index)

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeatureMatrixType):
            return False
        if self._name != feature_type._name:
            return False
        return True

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

    def to_dict(self):
        obj = dict()
        obj["type"] = "FeatureMatrixType"
        obj["name"] = self._name
        obj["matrix_fn"] = pickle.dumps(self._matrix_fn)
        obj["size"] = self._size

        if self._index is not None:
            obj["index"] = self._index

        return obj

    def save(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeatureMatrixType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        matrix_fn = pickle.loads(obj["matrix_fn"])
        size = obj["size"]
        index = None
        if "index" in obj:
            index = obj["index"]

        return FeatureMatrixType(name, matrix_fn, size, index=index)

class FeatureBucketedValueType(FeatureMatrixType):
    def __init__(self, name, value_fn, bucket_max, bucket_width, bucket_min=0.0):
        FeatureMatrixType.__init__(self, name, lambda d : self._make_bucket_one_hot(value_fn(d)), int((bucket_max-bucket_min)/bucket_width))
        self._value_fn = value_fn
        self._bucket_min = bucket_min
        self._bucket_max = bucket_max
        self._bucket_width = bucket_width

    def _make_bucket_one_hot(self, value):
        one_hot = np.zeros(self._size)
        if isinstance(value, np.ndarray):
            indices = np.floor((value-self._bucket_min)/self._bucket_width)
            for index in indices:
                index = int(min(index, len(one_hot) - 1))
                one_hot[index] += 1.0
            one_hot /= max(1.0, len(indices))
        else:
            index = int(min(np.floor((value-self._bucket_min)/self._bucket_width), len(one_hot) - 1))
            one_hot[index] = 1.0
        return one_hot

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeatureBucketedValueType):
            return False
        if self._name != feature_type._name:
            return False
        return True

    def to_dict(self):
        obj = dict()
        obj["type"] = "FeatureBucketedValueType"
        obj["name"] = self._name
        obj["value_fn"] = pickle.dumps(self._value_fn)
        obj["bucket_min"] = self._bucket_min
        obj["bucket_max"] = self._bucket_max
        obj["bucket_width"] = self._bucket_width
        return obj

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeatureBucketedValueType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        value_fn = pickle.loads(obj["value_fn"])
        bucket_min = obj["bucket_min"]
        bucket_max = obj["bucket_max"]
        bucket_width = obj["bucket_width"]
        return FeatureBucketedValueType(name, value_fn, bucket_max, bucket_width, bucket_min=bucket_min)


class FeatureMatrixSequence(FeatureSequence):
    def __init__(self, name, matrix_fn, length_fn, sequence_length, feature_size):
        FeatureSequence.__init__(self)
        self._name = name
        self._length_fn = length_fn
        self._matrix_fn = matrix_fn
        self._sequence_length = sequence_length
        self._feature_size = feature_size

        self._types = []
        for i in range(self._sequence_length):
            def vector_fn(datum, index):
                mat = matrix_fn(datum)
                if index >= mat.shape[0]:
                    return np.zeros(self._feature_size)
                else:
                    return mat[index]

            self._types.append(FeatureMatrixType(self._name + "." + str(i), vector_fn, self._feature_size, index=i))

    def __eq__(self, feature_seq):
        if not isinstance(feature_seq, FeatureMatrixSequence):
            return False
        if self._name != feature_seq._name:
            return False
        return True

    def get_datum_length(self, datum):
        return self._length_fn(datum)

    def get_name(self):
        return self._name

    def get_size(self):
        return self._sequence_length

    def get_type(self, index):
        return self._types[index]

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeatureMatrixSequence"
        obj["name"] = self._name
        obj["matrix_fn"] = pickle.dumps(self._matrix_fn)
        obj["length_fn"] = pickle.dumps(self._length_fn)
        obj["sequence_length"] = self._sequence_length
        obj["feature_size"] = self._feature_size
        with open(file_path, 'wb') as fp:
            pickle.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeatureMatrixSequence.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        matrix_fn = pickle.loads(obj["matrix_fn"])
        length_fn = pickle.loads(obj["length_fn"])
        sequence_length = obj["sequence_length"]
        feature_size = obj["feature_size"]
        return FeatureMatrixSequence(name, matrix_fn, length_fn, sequence_length, feature_size)


class FeaturePathToken(FeatureToken):
    def __init__(self, name, path, index, token):
        FeatureToken.__init__(self)
        self._name = name
        self._path = path
        self._index = index
        self._token = token

    def __str__(self):
        return self._name + "_" + self._path + "_" + str(self._token)

    def get_name(self):
        return self._name + "_" + self._path

    def get_path(self):
        return self._path

    def get_value(self):
        return self._token

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

class FeaturePathType(FeatureType):
    def __init__(self, name, paths, fixed_path_values=None, min_occur=1, no_init=False, value_type=ValueType.ENUMERABLE_ONE_HOT, value_fn=None, seq_index=None, vocab=None, token_fn=lambda x : x, ignore_list_indices=False):
        FeatureType.__init__(self)
        self._name = name
        self._paths = paths
        self._min_occur = min_occur
        self._no_init = no_init
        self._value_type = value_type
        self._value_fn = value_fn
        self._token_fn = token_fn
        self._seq_index = seq_index

        self._counter = None
        self._vocab = vocab
        self._fixed_path_values = fixed_path_values
        self._ignore_list_indices = ignore_list_indices

    def get_datum_length(self, datum):
        if self._seq_index is None:
            return 0

        seq = self._get_path_values(datum)
        for i in range(len(seq)):
            if len(seq[i]) == 0:
                return i
        return len(seq)

    # Map (path -> values) to (key, value) list list
    # representing a sequence of mappings or (key, value) list
    # represnting a single mapping
    def _apply_value_fn(self, path_to_values):
        if self._value_fn is not None:
            return self._value_fn(path_to_values)

        if self._seq_index is not None:
            return self._apply_value_fn_seq(path_to_values)
        else:
            return self._apply_value_fn_nonseq(path_to_values)

    def _apply_value_fn_seq(self, path_to_values):
        seq = [[] for i in range(self._seq_index+1)]

        for path in path_to_values:
            values = path_to_values[path]
            if len(values) == 0:
                values = [self._token_fn(Symbol.SEQ_START), self._token_fn(Symbol.SEQ_END)]
            elif isinstance(values[0], list):
                new_values = [self._token_fn(Symbol.SEQ_START)]
                for value in values:
                    new_values.extend([self._token_fn(el) for el in value])
                    new_values.append(self._token_fn(Symbol.SEQ_MID))
                new_values[len(new_values)-1] = self._token_fn(Symbol.SEQ_END)
                values = new_values[:self._seq_index+1]
            else:
                new_values = [self._token_fn(Symbol.SEQ_START)]
                new_values.extend([self._token_fn(value) for value in values])
                new_values.append(self._token_fn(Symbol.SEQ_END))
                values = new_values[:self._seq_index+1]

            for i in range(min(len(values), len(seq))):
                seq[i].append((path, values[i]))
        return seq

    def _apply_value_fn_nonseq(self, path_to_values):
        mapping = []

        for path in path_to_values:
            values = path_to_values[path]
            if len(values) == 0:
                continue
            if isinstance(values[0], list):
                values = [el for value in values for el in value]
            for i in range(len(values)):
                if self._ignore_list_indices:
                    mapping.append((path, self._token_fn(values[i])))
                else:
                    mapping.append((path + "_" + str(i), self._token_fn(values[i])))
        return mapping

    # Get sequence of (key -> value) mappings represented as
    # (key,value) lists or just a single (key, value) list if
    # not sequential
    def _get_path_values(self, datum):
        path_to_values = dict()
        for path in self._paths:
            values = datum.get(path, first=False)
            path_to_values[path] = values
        return self._apply_value_fn(path_to_values)

    def get_name(self):
        return self._name

    def compute(self, datum, vec, start_index):
        path_values = self._get_path_values(datum)
        if self._seq_index is not None:
            path_values = path_values[self._seq_index]

        for path_value in path_values:
            index = None
            value = None
            if self._value_type == ValueType.SCALAR:
                index = self._vocab[path_value[0]]
                vec[start_index + index] = path_value[1]
            else:
                key = path_value[0] + "__" + path_value[1]
                if key in self._vocab:
                    index = self._vocab[key]
                elif path_value[0] + "__" + self._token_fn(Symbol.SEQ_UNC) in self._vocab:
                    index = self._vocab[path_value[0] + "__" + self._token_fn(Symbol.SEQ_UNC)]
                else:
                    continue

                if self._value_type == ValueType.ENUMERABLE_INDEX:
                    vec[start_index] = index
                else:
                    vec[start_index + index] = 1.0

    def get_token_count(self):
        return len(self._vocab)

    def get_size(self):
        if self._value_type == ValueType.ENUMERABLE_INDEX:
            return 1
        else:
            return len(self._vocab)

    def get_token(self, index):
        # FIXME This is a bit of a hack (splitting for convenience)
        path_value = self._vocab.inv[index]
        under_idx = path_value.rfind("__")
        path = path_value[:under_idx]
        value = ""
        if under_idx < len(path_value) - 1:
            value = path_value[under_idx+2:]

        return FeaturePathToken(self._name, path, index, value)

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeaturePathType):
            return False
        if self._name != feature_type._name:
            return False
        return True

    def init_start(self):
        if self._no_init:
            return
        self._counter = Counter()

    def init_datum(self, datum):
        if self._no_init:
            return
        if self._seq_index is not None:
            path_values_seq = self._get_path_values(datum)
            if self._value_type == ValueType.SCALAR:
                for path_values in path_values_seq:
                    for path_value in path_values:
                        self._counter[path_value[0]] += 1
            else:
                for path_values in path_values_seq:
                    for path_value in path_values:
                        self._counter[path_value[0] + "__" + path_value[1]] += 1
        else:
            path_values = self._get_path_values(datum)
            if self._value_type == ValueType.SCALAR:
                for path_value in path_values:
                    self._counter[path_value[0]] += 1
            else:
                for path_value in path_values:
                    self._counter[path_value[0] + "__" + path_value[1]] += 1

    def init_end(self):
        if self._no_init:
            return

        if self._fixed_path_values is not None:
            self._vocab = bidict()
            index = 0
            for path, values in self._fixed_path_values.items():
                for value in values:
                    path_value = path + "__" + self._token_fn(value)
                    if path_value not in self._vocab:
                        self._vocab[path_value] = index
                        index += 1
            self._counter = None
            return
        
        vocab_list = []
        for key in self._counter:
            if self._counter[key] >= self._min_occur:
                vocab_list.append(key)

        if self._value_type != ValueType.SCALAR:
            vocab_list.sort()
        else:
            # If scalar, then keep dimensions in the order they were given
            def cmp(v1, v2):
                i1 = 0
                i2 = 0

                v1_path = v1[0:v1.rfind("__")]
                v2_path = v2[0:v2.rfind("__")]

                for i in range(len(self._paths)):
                    if self._paths[i] == v1_path:
                        i1 = i
                    if self._paths[i] == v2_path:
                        i2 = i
                if i1 < i2:
                    return -1
                elif i1 > i2:
                    return 1
                else:
                    if v1 < v2:
                        return -1
                    elif v2 < v1:
                        return 1
                    else:
                        return 0

            vocab_list.sort(key=cmp_to_key(cmp))


        if self._vocab is None:
            self._vocab = bidict()

        index = 0
        if self._seq_index is not None and self._value_type != ValueType.SCALAR:
            for path in self._paths:
                self._vocab[path + "__" + self._token_fn(Symbol.SEQ_UNC)] = index*4 + Symbol.index(Symbol.SEQ_UNC)
                self._vocab[path + "__" + self._token_fn(Symbol.SEQ_START)] = index*4 + Symbol.index(Symbol.SEQ_START)
                self._vocab[path + "__" + self._token_fn(Symbol.SEQ_END)] = index*4 + Symbol.index(Symbol.SEQ_END)
                self._vocab[path + "__" + self._token_fn(Symbol.SEQ_MID)] = index*4 + Symbol.index(Symbol.SEQ_MID)
                index += 1

            index = index*4
            for v in vocab_list:
                if v not in self._vocab:
                    self._vocab[v] = index
                    index += 1
        else:
            for v in vocab_list:
                if v not in self._vocab:
                    self._vocab[v] = index
                    index += 1

        self._counter = None

    def to_dict(self):
        obj = dict()
        obj["type"] = "FeaturePathType"
        obj["name"] = self._name
        obj["paths"] = self._paths
        obj["min_occur"] = self._min_occur
        obj["no_init"] = self._no_init
        obj["value_type"] = self._value_type
        if self._value_fn is not None:
            obj["value_fn"] = pickle.dumps(self._value_fn)
        if self._token_fn is not None:
            obj["token_fn"] = pickle.dumps(self._token_fn)
        if self._seq_index is not None:
            obj["seq_index"] = self._seq_index
        if self._vocab is not None:
            obj["vocab"] = dict(self._vocab)

        obj["ignore_list_indices"] = self._ignore_list_indices

        return obj

    def save(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeaturePathType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        paths = obj["paths"]
        min_occur = obj["min_occur"]
        no_init = obj["no_init"]
        value_type = obj["value_type"]
        value_fn = None
        if "value_fn" in obj:
            value_fn = pickle.loads(obj["value_fn"])
        token_fn = None
        if "token_fn" in obj:
            token_fn = pickle.loads(obj["token_fn"])
        seq_index = None
        if "seq_index" in obj:
            seq_index = obj["seq_index"]
        vocab = None
        if "vocab" in obj:
            vocab = bidict(obj["vocab"])
        ignore_list_indices = False
        if "ignore_list_indices" in obj:
            ignore_list_indices = obj["ignore_list_indices"]

        return FeaturePathType(name, paths, min_occur=min_occur, no_init=no_init, value_type=value_type, value_fn=value_fn, seq_index=seq_index, token_fn=token_fn, vocab=vocab, ignore_list_indices=ignore_list_indices)


class FeaturePathSequence(FeatureSequence):
    def __init__(self, name, paths, seq_length, min_occur=2, value_type=ValueType.ENUMERABLE_ONE_HOT, value_fn=None, token_fn=None, vocab=None):
        FeatureSequence.__init__(self)
        self._name = name
        self._paths = paths
        self._seq_length = seq_length
        self._min_occur = min_occur
        self._value_type = value_type
        self._value_fn = value_fn
        self._token_fn = token_fn

        self._vocab = None
        self._init_type = None
        self._types = []

        self._vocab = vocab
        self._init_types()

    def _init_types(self):
        self._types = []
        for i in range(self._seq_length):
            self._types.append(FeaturePathType(self._name,
                self._paths,
                min_occur=self._min_occur,
                no_init=True,
                value_type=self._value_type,
                value_fn=self._value_fn,
                seq_index=i,
                token_fn=self._token_fn,
                vocab=self._vocab
            ))

    def __eq__(self, feature_seq):
        if not isinstance(feature_seq, FeaturePathSequence):
            return False
        if self._name != feature_seq._name:
            return False
        return True

    def get_name(self):
        return self._name

    def get_size(self):
        return self._seq_length

    def get_datum_length(self, datum):
        return self._types[len(self._types) - 1].get_datum_length(datum)

    def get_type(self, index):
        return self._types[index]

    def init_start(self):
        self._vocab = bidict()
        self._init_type = FeaturePathType(
            self._name,
            self._paths,
            min_occur=self._min_occur,
            no_init=False,
            value_type=self._value_type,
            value_fn=self._value_fn,
            seq_index=self._seq_length-1,
            token_fn=self._token_fn,
            vocab=self._vocab)
        self._init_type.init_start()

    def init_datum(self, datum):
        self._init_type.init_datum(datum)

    def init_end(self):
        self._init_type.init_end()
        self._init_types()
        self._init_type = None

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeaturePathSequence"
        obj["name"] = self._name
        obj["paths"] = self._paths
        obj["seq_length"] = self._seq_length
        obj["min_occur"] = self._min_occur
        obj["value_type"] = self._value_type
        if self._value_fn is not None:
            obj["value_fn"] = pickle.dumps(self._value_fn)
        if self._token_fn is not None:
            obj["token_fn"] = pickle.dumps(self._token_fn)
        if self._vocab is not None:
            obj["vocab"] = dict(self._vocab)

        with open(file_path, 'wb') as fp:
            pickle.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeaturePathSequence.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        paths = obj["paths"]
        seq_length = obj["seq_length"]
        min_occur = obj["min_occur"]
        value_type = obj["value_type"]
        value_fn = None
        if "value_fn" in obj:
            value_fn = pickle.loads(obj["value_fn"])
        token_fn = None
        if "token_fn" in obj:
            token_fn = pickle.loads(obj["token_fn"])
        vocab = None
        if "vocab" in obj:
            vocab = bidict(obj["vocab"])
        return FeaturePathSequence(name, paths, seq_length, min_occur=min_occur, value_type=value_type, value_fn=value_fn, token_fn=token_fn, vocab=vocab)

class FeaturePathVectorDictionaryType(FeaturePathType):
    def __init__(self, name, paths, vector_dict, vector_fn=lambda v : v, vocab=None, token_fn=lambda x : x, internalize_dict=True):
        FeaturePathType.__init__(self, name, paths, value_type=ValueType.SCALAR, value_fn=self._value_fn, token_fn=token_fn, vocab=vocab)
        self._vector_dict = vector_dict
        self._vector_fn = vector_fn
        self._internalize_dict = internalize_dict

    def __eq__(self, feature_type):
        return isinstance(feature_type, FeaturePathVectorDictionaryType) and self._name == feature_type._name

    def _value_fn(self, path_to_values):
        mapping = []
        for path in path_to_values:
            values = path_to_values[path]
            if len(values) == 0:
                continue
            if isinstance(values[0], list):
                values = [el for value in values for el in value]
            for i in range(len(values)):
                transformed_value = self._token_fn(values[i])
                if transformed_value in self._vector_dict:
                    vec = self._vector_dict[transformed_value]
                else:
                    vec = self._vector_dict.get_default_vector()

                vec = self._vector_fn(vec)

                for j in range(len(vec)):
                    mapping.append((path + "_" + str(i) + "_" + str(j), vec[j]))
        return mapping

    def to_dict(self):
        # Same as FeaturePathType... should probably be changed if FeaturePathType changes
        # Also, should eventually FIXME to remove the redundancy
        obj = dict()
        obj["type"] = "FeaturePathVectorDictionaryType"
        obj["name"] = self._name
        obj["paths"] = self._paths
        obj["min_occur"] = self._min_occur
        obj["no_init"] = self._no_init
        obj["value_type"] = self._value_type
        if self._token_fn is not None:
            obj["token_fn"] = pickle.dumps(self._token_fn)
        if self._seq_index is not None:
            obj["seq_index"] = self._seq_index
        if self._vocab is not None:
            obj["vocab"] = dict(self._vocab)

        # Not FeaturePathType... specific to VectorDictionaryType
        obj["internalize_dict"] = self._internalize_dict
        if self._internalize_dict:
            obj["vector_dict"] = self._vector_dict
        if self._vector_fn is not None:
            obj["vector_fn"] = pickle.dumps(self._vector_fn)

        return obj

    def save(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeaturePathVectorDictionaryType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        # Same as FeaturePathType... should probably be changed if FeaturePathType changes
        # Also, should eventually FIXME to remove the redundancy
        name = obj["name"]
        paths = obj["paths"]
        
        # Unnecessary for now 
        #min_occur = obj["min_occur"]
        #no_init = obj["no_init"]
        #value_type = obj["value_type"]
        #value_fn = None
        #if "value_fn" in obj:
        #    value_fn = pickle.loads(obj["value_fn"])
        token_fn = None
        if "token_fn" in obj:
            token_fn = pickle.loads(obj["token_fn"])
        #seq_index = None
        #if "seq_index" in obj:
        #    seq_index = obj["seq_index"]
        vocab = None
        if "vocab" in obj:
            vocab = bidict(obj["vocab"])

        # Not FeaturePathType... specific to VectorDictionaryType
        internalize_dict = obj["internalize_dict"]
        if internalize_dict:
            vector_dict = obj["vector_dict"]
        else:
            vector_dict = StoredVectorDictionary(vecs=dict())

        vector_fn = None
        if "vector_fn" in obj:
            vector_fn = pickle.loads(obj["vector_fn"])

        return FeaturePathVectorDictionaryType(name, paths, vector_dict, token_fn=token_fn, vector_fn=vector_fn, vocab=vocab, internalize_dict=internalize_dict)

class FeatureProductToken(FeatureToken):
    def __init__(self, name, token_0, token_1):
        FeatureToken.__init__(self)
        self._name = name
        self._token_0 = token_0
        self._token_1 = token_1

    def __str__(self):
        return self._name + "_" + str(self._token_0) + "_" + str(self._token_1)

    def get_name(self):
        return self._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

class FeatureProductType(FeatureType):
    def __init__(self, name, feature_0, feature_1, no_init=False):
        FeatureType.__init__(self)
        self._name = name
        self._no_init = no_init
        self._feature_0 = feature_0
        self._feature_1 = feature_1

    def get_name(self):
        return self._name

    def get_size(self):
        return self._feature_0.get_size()*self._feature_1.get_size()

    def compute(self, datum, vec, start_index):
        vec_0 = np.zeros(shape=self._feature_0.get_size())
        vec_1 = np.zeros(shape=self._feature_1.get_size())

        self._feature_0.compute(datum, vec_0, 0)
        self._feature_1.compute(datum, vec_1, 0)

        for i in range(self._feature_0.get_size()):
            for j in range(self._feature_1.get_size()):
                index = i*self._feature_1.get_size() + j
                vec[start_index + index] = vec_0[i]*vec_1[j]

    def get_token(self, index):
        index_0 = index / self._feature_1.get_size()
        index_1 = index % self._feature_1.get_size()
        return FeatureProductToken(self._name, self._feature_0.get_token(index_0), self._feature_1.get_token(index_1))

    def __eq__(self, feature_type):
        return isinstance(feature_type, FeatureProductType) and self._name == feature_type._name

    def init_start(self):
        if self._no_init:
            return
        self._feature_0.init_start()
        self._feature_1.init_start()

    def init_datum(self, datum):
        if self._no_init:
            return
        self._feature_0.init_datum(datum)
        self._feature_1.init_datum(datum)

    def init_end(self):
        if self._no_init:
            return
        self._feature_0.init_end()
        self._feature_1.init_end()

    def to_dict(self):
        obj = dict()
        obj["type"] = "FeatureProductType"
        obj["name"] = self._name
        obj["feature_0"] = self._feature_0.to_dict()
        obj["feature_1"] = self._feature_1.to_dict()
        obj["no_init"] = self._no_init
        return obj

    def save(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.to_dict(), fp)

    @staticmethod
    def from_dict(obj):
        feature_0 = None
        if obj["feature_0"]["type"] in FEATURE_TYPES:
            feature_0 = FEATURE_TYPES[obj["feature_0"]["type"]].from_dict(obj["feature_0"])
        else:
            raise ValueError(obj["feature_0"]["type"] + " feature type not registered")

        feature_1 = None
        if obj["feature_1"]["type"] in FEATURE_TYPES:
            feature_1 = FEATURE_TYPES[obj["feature_1"]["type"]].from_dict(obj["feature_1"])
        else:
            raise ValueError(obj["feature_1"]["type"] + " feature type not registered")

        return FeatureProductType(obj["name"], feature_0, feature_1, no_init=obj["no_init"])

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as fp:
            obj = pickle.load(fp)
            return FeatureProductType.from_dict(obj)

class FeatureSet:
    def __init__(self, feature_types=[]):
        self._feature_types = list(feature_types)

    def has_feature_type(self, feature_type):
        for f in self._feature_types:
            if f == feature_type:
                return True
        return False

    def add_feature_type(self, feature_type):
        if self.has_feature_type(feature_type):
            return False
        self._feature_types.append(feature_type)
        return True

    def add_feature_types(self, feature_types):
        ret = []
        for feature_type in feature_types:
            r = self.add_feature_type(feature_type)
            if r:
                ret.append(feature_type)
        return ret

    def get_size(self):
        return sum([feature_type.get_size() for feature_type in self._feature_types])

    def get_token_count(self):
        token_count = 0
        for feature_type in self._feature_types:
            token_count += feature_type.get_token_count()
        return token_count

    def get_num_feature_types(self):
        return len(self._feature_types)

    def compute(self, datum, start_from=0, v=None):
        if v is None:
            v = np.array(self.get_size())
        start_index = 0
        for i in range(0, len(self._feature_types)):
            if i >= start_from:
                self._feature_types[i].compute(datum, v, start_index)
            start_index += self._feature_types[i].get_size()
        return v

    def init(self, data):
        self.init_start()
        for i in range(data.get_size()):
            self.init_datum(data.get(i))
        self.init_end()

    def init_start(self, start_from=0):
        for i in range(start_from, len(self._feature_types)):
            self._feature_types[i].init_start()

    def init_datum(self, datum, start_from=0):
        for i in range(start_from, len(self._feature_types)):
            self._feature_types[i].init_datum(datum)

    def init_end(self, start_from=0):
        for i in range(start_from, len(self._feature_types)):
            self._feature_types[i].init_end()

    def get_feature_token(self, index):
        offset = 0
        for i in range(len(self._feature_types)):
            if index < offset + self._feature_types[i].get_token_count():
                return self._feature_types[i].get_token(index - offset)
            else:
                offset += self._feature_types[i].get_token_count()
        return None

    def get_feature_type(self, index):
        return self._feature_types[index]

    def copy(self):
        return FeatureSet(feature_types=self._feature_types)

    def save(self, dir_path):
        info = dict()
        info["feats"] = [feature_type.get_name() for feature_type in self._feature_types]

        info_path = join(dir_path, "info")
        with open(info_path, 'w') as fp:
            json.dump(info, fp)

        for feature_type in self._feature_types:
            feature_type.save(join(dir_path, feature_type.get_name()))

    def make_token_lookup(self):
        lookup = dict()
        for i in range(self.get_token_count()):
            lookup[self.get_feature_token(i).get_value()] = i
        return lookup

    @staticmethod
    def load(dir_path):
        info_path = join(dir_path, "info")
        info = None
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        file_paths = [join(dir_path, f) for f in info["feats"]]
        
        feature_types = []
        for file_path in file_paths:
            with open(file_path, 'rb') as fp:
                obj = pickle.load(fp)
                if obj["type"] in FEATURE_TYPES:
                    feature_types.append(FEATURE_TYPES[obj["type"]].from_dict(obj))
                else:
                    raise ValueError(obj["type"] + " feature type not registered")
        return FeatureSet(feature_types=feature_types)


class FeatureSequenceSet:
    def __init__(self, feature_seqs=[]):
        self._feature_sets = []
        self._feature_seqs = list(feature_seqs)
        if len(feature_seqs) > 0:
            for feature_seq in feature_seqs:
                if feature_seq.get_size() != feature_seqs[0].get_size():
                    raise ValueError("Feature sequences in FeatureSequenceSet must have the same size.")
        self._init_feature_sets()

    def _init_feature_sets(self):
        self._feature_sets = []
        for i in range(self._feature_seqs[0].get_size()):
            feature_set = FeatureSet(feature_types=[feature_seq.get_type(i) for feature_seq in self._feature_seqs])
            self._feature_sets.append(feature_set)

    def get_datum_max_length(self, datum):
        max_len = 0
        for feature_seq in self._feature_seqs:
            max_len = max(max_len, feature_seq.get_datum_length(datum))
        return max_len

    def has_feature_seq(self, feature_seq):
        for f in self._feature_seqs:
            if f == feature_seq:
                return True
        return False

    def add_feature_seq(self, feature_seq):
        if len(self._feature_seqs) > 0 and feature_seq.get_size() != self._feature_seqs[0].get_size():
            raise ValueError("Feature sequences in FeatureSequenceSet must have the same size.")

        if self.has_feature_seq(feature_seq):
            return False
        self._feature_seqs.append(feature_seq)

        if len(self._feature_seqs) == 1:
            self._init_feature_sets()
        else:
            for i in range(self._feature_seqs[0].get_size()):
                self._feature_sets[i].add_feature_type(feature_seq.get_type(i))

        return True

    def add_feature_seqs(self, feature_seqs):
        ret = []
        for feature_seq in feature_seqs:
            r = self.add_feature_seq(feature_seq)
            if r:
                ret.append(feature_seq)
        return ret

    def get_size(self):
        if len(self._feature_seqs) == 0:
            return 0
        else:
            return self._feature_seqs[0].get_size()

    def get_feature_set_size(self):
        if len(self._feature_sets) == 0:
            return 0
        else:
            return self._feature_sets[0].get_size()

    def get_num_feature_seqs(self):
        return len(self._feature_seqs)

    def init(self, data):
        self.init_start()
        for i in range(data.get_size()):
            self.init_datum(data.get(i))
        self.init_end()

    def init_start(self, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_start()

    def init_datum(self, datum, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_datum(datum)

    def init_end(self, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_end()
        self._init_feature_sets()

    def get_feature_token(self, index, seq_index=0):
        if len(self._feature_sets) == 0:
            return None
        return self._feature_sets[seq_index].get_feature_token(index)

    def get_feature_seq(self, index):
        return self._feature_seqs[index]

    def get_feature_set(self, index=0):
        return self._feature_sets[index]

    def copy(self):
        return FeatureSequenceSet(feature_seqs=self._feature_seqs)

    def save(self, dir_path):
        for feature_seq in self._feature_seqs:
            feature_seq.save(join(dir_path, feature_seq.get_name()))

    @staticmethod
    def load(dir_path):
        file_paths = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
        feature_seqs = []
        for file_path in file_paths:
            with open(file_path, 'rb') as fp:
                obj = pickle.load(fp)
                if obj["type"] in FEATURE_SEQ_TYPES:
                    feature_seqs.append(FEATURE_SEQ_TYPES[obj["type"]].from_dict(obj))
                else:
                    raise ValueError(obj["type"] + " feature sequence type not registered")
        return FeatureSequenceSet(feature_seqs=feature_seqs)


class DataFeatureMatrix:
    def __init__(self, data, feature_set, init_features=True, mat=None, compute_non_zero=False):
        self._data = data
        self._feature_set = feature_set
        self._mat = None
        self._nz_indices = None
        self._compute_nz = compute_non_zero
        if mat is None:
            self._compute(init_features=init_features)
        else:
            self._mat = mat
            if self._compute_nz:
                self._nz_indices = []
                for i in range(data.get_size()):
                    self._nz_indices.append([])
                    for j in range(len(self._mat[i])):
                        if self._mat[i,j] != 0.0:
                            self._nz_indices[i].append(j)

    def get_data(self):
        return self._data

    def get_feature_set(self):
        return self._feature_set

    def get_feature_token(self, index):
        return self._feature_set.get_feature_token(index)

    def get_matrix(self):
        return self._mat

    def get_vector(self, i):
        return self._mat[i]

    def get_batch(self, i, size, form=ArrayFormat.TORCH):
        if size > self.get_data().get_size():
            raise ValueError("Batch size cannot be greater than data set size")

        return ArrayFormat.cast(self._mat[i*size:(i+1)*size], form)

    def get_final_batch(self, size, form=ArrayFormat.TORCH):
        if size > self.get_data().get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        final_size = self._data.get_size() % size
        if final_size == 0:
            return None
        return ArrayFormat.cast(self._mat[self._data.get_size()-final_size:self._data.get_size()], form)

    def get_num_batches(self, size):
        if size > self.get_data().get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self._data.get_size() // size

    def get_random_batch(self, size, form=ArrayFormat.TORCH):
        if size > self.get_data().get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        batch_indices = np.random.choice(self.get_size(), size, replace=False)
        return ArrayFormat.cast(self._mat[batch_indices], form)

    def get_batch_by_indices(self, batch_indices, form=ArrayFormat.TORCH):
        return ArrayFormat.cast(self._mat[batch_indices], form)

    def get_non_zero_indices(self, i):
        if self._compute_nz:
            return self._nz_indices[i]
        else:
            return None

    def extend(self, feature_types, start_num=None, start_size=None):
        if start_num is None:
            start_num = self._feature_set.get_num_feature_types()
            start_size = self._feature_set.get_size()
            added = self._feature_set.add_feature_types(feature_types)

            if len(added) == 0:
                return

            #print "Extending feature matrix... (" + str(self._feature_set.get_size()-start_size) + " feature tokens)"

            self._feature_set.init_start(start_from=start_num)
            for i in range(self._data.get_size()):
                self._feature_set.init_datum(self._data.get(i), start_from=start_num)
            self._feature_set.init_end(start_from=start_num)

        ext_mat = np.zeros((self._data.get_size(), self._feature_set.get_size()-start_size))
        self._mat = np.concatenate((self._mat, ext_mat), axis=1)

        for i in range(self._data.get_size()):
            self._feature_set.compute(self._data.get(i), start_from=start_num, v=self._mat[i])

            if self._compute_nz:
                nz = []
                for j in range(start_size, len(self._mat[i])):
                    if self._mat[i,j] != 0:
                        nz.append(j)
                self._nz_indices[i].extend(nz)

    def _compute(self, init_features=True):
        self._mat = np.zeros((self._data.get_size(), self._feature_set.get_size()))
        self._nz_indices = []

        if init_features:
            self._feature_set.init_start()
            for i in range(self._data.get_size()):
                self._feature_set.init_datum(self._data.get(i))
            self._feature_set.init_end()

        #print "Computing feature matrix... (" + str(self._feature_set.get_size()) + " features)"
        for i in range(self._data.get_size()):
            self._feature_set.compute(self._data.get(i), v=self._mat[i])

            if self._compute_nz:
                nz = []
                for j in range(self._feature_set.get_size()):
                    if self._mat[i][j] != 0.0:
                        nz.append(j)
                self._nz_indices.append(nz)
        #print "Finished computing matrix"

    def shuffle(self):
        perm = np.random.permutation(len(self._mat))
        self.reorder(perm)

    def reorder(self, perm, preordered_data=None):
        shuffled_mat = np.zeros(self._mat.shape)
        shuffled_nz = []
        shuffled_data = []
        for i in range(len(perm)):
            np.copyto(shuffled_mat[i], self._mat[perm[i]])
            if self._compute_nz:
                shuffled_nz.append(self._nz_indices[perm[i]])
            shuffled_data.append(self._data.get(perm[i]))

        if preordered_data is None:
            self._data = DataSet(data=shuffled_data)
        else:
            self._data = preordered_data

        self._mat = shuffled_mat
        if self._compute_nz:
            self._nz_indices = shuffled_nz

    def partition(self, partition, key_fn):
        data_parts = self._data.partition(partition, key_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i
        return self._data_partition(data_parts, id_to_index, key_fn)

    def _data_partition(self, data_parts, id_to_index, key_fn):
        dfmat_parts = dict()

        for key, data_part in data_parts.items():
            mat_part = np.zeros([data_part.get_size(), self._feature_set.get_size()])
            for i in range(data_part.get_size()):
                mat_part[i] = self._mat[id_to_index[data_part.get(i).get_id()]]
            dfmat_parts[key] = DataFeatureMatrix(data_part, self._feature_set, init_features=False, mat=mat_part, compute_non_zero=self._compute_nz)
        return dfmat_parts

    def union(self, other, data=None):
        data_union = None
        if data is not None:
            data_union = data
        else:
            data = self._data.union(other._data)

        mat_union = np.concatenate((self._mat, other._mat), axis=0)
        return DataFeatureMatrix(data_union, self._feature_set, init_features=False, mat=mat_union, compute_non_zero=False)

    def filter(self, filter_fn):
        filtered_data = self._data.filter(filter_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i
        return self._data_filter(filtered_data, id_to_index, filter_fn)

    def _data_filter(self, filtered_data, id_to_index, filter_fn):
        mat_filtered = np.zeros([filtered_data.get_size(), self._feature_set.get_size()])
        for i in range(filtered_data.get_size()):
            mat_filtered[i] = self._mat[id_to_index[filtered_data.get(i).get_id()]]
        return DataFeatureMatrix(filtered_data, self._feature_set, init_features=False, mat=mat_filtered, compute_non_zero=self._compute_nz)

    def save(self, dir_path):
        info_path = join(dir_path, "info")
        mat_path = join(dir_path, "mat")
        feats_dir = join(dir_path, "feats")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not os.path.exists(feats_dir):
            os.makedirs(feats_dir)
        else:
            print("Warning: " + feats_dir + " already exists... overwrite may be unsafe.  Delete manually.")

        data_order = [self._data.get(i).get(self._data.get_id_key()) for i in range(self._data.get_size())]

        info = dict()
        info["data_dir"] = self._data.get_directory()
        info["id_key"] = self._data.get_id_key()
        info["compute_nz"] = self._compute_nz
        info["data_order"] = data_order

        with open(info_path, 'w') as fp:
            json.dump(info, fp)

        np.save(mat_path, self._mat)
        self._feature_set.save(feats_dir)

    @staticmethod
    def load(dir_path, data=None, data_cls=None):
        info_path = join(dir_path, "info")
        mat_path = join(dir_path, "mat.npy")
        feats_dir = join(dir_path, "feats")

        mat = np.load(mat_path)
        feature_set = FeatureSet.load(feats_dir)

        obj = None
        with open(info_path, 'r') as fp:
            obj = json.load(fp)

        if data is None:
            if data_cls is None:
                data_cls = DataSet
            data = data_cls.load(obj["data_dir"], id_key=obj["id_key"], order=obj["data_order"])
            return DataFeatureMatrix(data, feature_set, init_features=False, mat=mat, compute_non_zero=obj["compute_nz"])
        else:
            mat_id_to_index = dict()
            for i in range(len(obj["data_order"])):
                mat_id_to_index[obj["data_order"][i]] = i
            # Perm : target index (data) -> source index (original mat)
            perm = [mat_id_to_index[data[i].get_id()] for i in range(data.get_size())]

            dfmat = DataFeatureMatrix(data, feature_set, init_features=False, mat=mat, compute_non_zero=obj["compute_nz"])
            dfmat.reorder(perm, preordered_data=data)
            return dfmat


class DataFeatureMatrixSequence:
    def __init__(self, data, feature_seq_set, init_features=True, mats=None, lengths=None, mask=None):
        self._data = data
        self._feature_seq_set = feature_seq_set
        self._dfmats = []
        self._lengths = lengths
        self._mask = mask

        if mats is None:
            self._compute(init_features=init_features)
        else:
            if lengths is None or mask is None:
                raise ValueError("If mats is supplied, then lengths and mask must also be supplied.")
            for i in range(len(mats)):
                self._dfmats.append(DataFeatureMatrix(data, self._feature_seq_set.get_feature_set(i), init_features=False, mat=mats[i]))

    def get_feature_token(self, index, seq_index=0):
        return self._feature_seq_set.get_feature_token(index, seq_index=seq_index)

    def get_data(self):
        return self._data

    def get_lengths_by_indices(self, indices):
        return self._lengths[indices]

    def get_feature_seq_set(self):
        return self._feature_seq_set

    def get_matrix(self, seq_i):
        return self._dfmats[seq_i]

    def get_vector(self, seq_i, data_i):
        return self._dfmats[seq_i].get_vector(data_i)

    # NOTE: These batch functions can probably be sped up quite a bit if necessary
    def get_random_batch(self, size, form=ArrayFormat.TORCH, sort_lengths=True, squeeze=True):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        batch_indices = np.random.choice(self.get_size(), size, replace=False)
        return self.get_batch_by_indices(batch_indices, form=form, sort_lengths=sort_lengths, squeeze=squeeze)

    def get_batch(self, batch_i, size, form=ArrayFormat.TORCH, sort_lengths=True, squeeze=True):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self.get_batch_by_indices(np.array(range(batch_i*size,(batch_i+1)*size)), form=form, sort_lengths=sort_lengths, squeeze=squeeze)

    def get_final_batch(self, size, form=ArrayFormat.TORCH, sort_lengths=True, squeeze=True):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        final_size = self._data.get_size() % size
        if final_size == 0:
            return None

        return self.get_batch_by_indices(np.array(range(self._data.get_size()-final_size,self._data.get_size())), form=form, sort_lengths=sort_lengths, squeeze=squeeze)

    def get_num_batches(self, size):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self._data.get_size() // size

    def get_batch_by_indices(self, batch_indices, form=ArrayFormat.TORCH, sort_lengths=True, squeeze=True):
        batch = np.zeros(shape=(len(self._dfmats),
                                len(batch_indices),
                                self._dfmats[0].get_feature_set().get_size()))

        lengths = self._lengths[batch_indices]
        if sort_lengths:
            bls = [(batch_indices, lengths) for (batch_indices,lengths)
                    in sorted(zip(batch_indices, lengths), key=lambda p: -p[1])]
            batch_indices = np.array([bl[0] for bl in bls])
            lengths = np.array([bl[1] for bl in bls])

        for i in range(len(self._dfmats)):
            batch[i] = self._dfmats[i].get_batch_by_indices(batch_indices, form=ArrayFormat.NUMPY)

        mask = self._mask[batch_indices]

        if squeeze:
            batch = np.squeeze(batch)

        return ArrayFormat.cast(batch, form), ArrayFormat.cast(lengths, form, ints=True), ArrayFormat.cast(mask, form)

    def extend(self, feature_seqs):
        start_num = self._feature_seq_set.get_num_feature_seqs()
        start_size = self._feature_seq_set.get_feature_set_size()
        added = self._feature_seq_set.add_feature_seqs(feature_seqs)

        if len(added) == 0:
            return

        self._feature_seq_set.init_start(start_from=start_num)
        for i in range(self._data.get_size()):
            self._feature_seq_set.init_datum(self._data.get(i), start_from=start_num)
        self._feature_seq_set.init_end(start_from=start_num)

        self._lengths = np.zeros(shape=(data.get_size()))
        self._mask = np.zeros(shape=(data.get_size(), self.get_size()))
        for i in range(self._data.get_size()):
            self._lengths[i] = self._feature_seq_set.get_datum_max_length(self._data.get(i))
            self._mask[i] = repeat([1,0],[self._lengths[i], self.get_size()-self._lengths[i]])

        for i in range(self._feature_seq_set.get_size()):
            feature_set = self._feature_seq_seq.get_feature_set(i)
            feature_set.extend(feature_seqs, start_num=start_num, start_size=start_size)

    def _compute(self, init_features=True):
        if init_features:
            self._feature_seq_set.init_start()
            for i in range(self._data.get_size()):
                self._feature_seq_set.init_datum(self._data.get(i))
            self._feature_seq_set.init_end()

        self._lengths = np.zeros(shape=(self._data.get_size()))
        self._mask = np.zeros(shape=(self._data.get_size(), self._feature_seq_set.get_size()))
        for i in range(self._data.get_size()):
            self._lengths[i] = self._feature_seq_set.get_datum_max_length(self._data.get(i))
            self._mask[i] = np.repeat([1,0],[self._lengths[i], self._feature_seq_set.get_size()-self._lengths[i]])

        for i in range(self._feature_seq_set.get_size()):
            feature_set = self._feature_seq_set.get_feature_set(i)
            self._dfmats.append(DataFeatureMatrix(self._data, feature_set, init_features=False))

    def shuffle(self):
        perm = np.random.permutation(self._data.get_size())
        self.reorder(perm)

    def reorder(self, perm, preordered_data=None):
        if preordered_data is None:
            shuffled_data = []
            shuffled_lengths = []
            shuffled_mask = []
            for i in range(len(perm)):
                shuffled_data.append(self._data.get(perm[i]))
                shuffled_lengths.append(self._lengths[perm[i]])
                shuffled_mask[i] = self._mask[perm[i]]
            self._data = DataSet(data=shuffled_data)
            self._lengths = np.array(shuffled_lengths)
            self._mask = shuffled_mask
        else:
            self._data = preordered_data
            self._lengths = np.zeros(shape=(self._data.get_size()))
            self._mask = np.zeros(shape=(self._data.get_size(), self._feature_seq_set.get_size()))
            for i in range(self._data.get_size()):
                self._lengths[i] = self._feature_seq_set.get_datum_max_length(self._data.get(i))
                self._mask[i] = np.repeat([1,0],[self._lengths[i], self._feature_seq_set.get_size()-self._lengths[i]])

        for dfmat in self._dfmats:
            dfmat.reorder(perm, preordered_data=self._data)

    def partition(self, partition, key_fn):
        data_parts = self._data.partition(partition, key_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i

        return self._data_partition(data_parts, id_to_index, key_fn)

    def _data_partition(self, data_parts, id_to_index, key_fn):
        dfmats_parts = dict()

        for key, data_part in data_parts.items():
            mats_part = [np.zeros(shape=(data_part.get_size(), self._feature_seq_set.get_feature_set(0).get_size())) for i in range(self._feature_seq_set.get_size())]
            part_indices = np.array([id_to_index[data_part.get(i).get_id()] for i in range(data_part.get_size())])
            lengths_part = self._lengths[part_indices]
            mask_part = self._mask[part_indices]
            for s in range(self._feature_seq_set.get_size()):
                for i in range(data_part.get_size()):
                    mats_part[s][i] = self._dfmats[s].get_matrix()[id_to_index[data_part.get(i).get_id()]]
            dfmats_parts[key] = DataFeatureMatrixSequence(data_part, self._feature_seq_set, mats=mats_part, lengths=lengths_part, mask=mask_part)

        return dfmats_parts

    def union(self, other, data=None):
        raise NotImplementedError("Union not yet implemented for sequences.")

    def filter(self, filter_fn):
        filtered_data = self._data.filter(filter_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i
        return self._data_filter(filtered_data, id_to_index, filter_fn)

    def _data_filter(self, data_filtered, id_to_index, filter_fn):
        mats_filtered = [np.zeros(shape=(data_filtered.get_size(), self._feature_seq_set.get_feature_set(0).get_size())) for i in range(self._feature_seq_set.get_size())]
        filtered_indices = np.array([id_to_index[data_filtered.get(i).get_id()] for i in range(data_filtered.get_size())])
        lengths_filtered = self._lengths[filtered_indices]
        mask_filtered = self._mask[filtered_indices]
        for s in range(self._feature_seq_set.get_size()):
            for i in range(data_filtered.get_size()):
                mats_filtered[s][i] = self._dfmats[s].get_matrix()[id_to_index[data_filtered.get(i).get_id()]]
        return DataFeatureMatrixSequence(data_filtered, self._feature_seq_set, mats=mats_filtered, lengths=lengths_filtered, mask=mask_filtered)

    def save(self, dir_path):
        info_path = join(dir_path, "info")
        mats_dir = join(dir_path, "mats")
        feats_dir = join(dir_path, "feats")
        mask_path = join(dir_path, "mask")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if os.path.exists(mats_dir) or os.path.exists(feats_dir):
            print("Warning: " + mats_dir + " or " + feats_dir + " already exists... overwrite may be unsafe.  Delete manually.")

        if not os.path.exists(mats_dir):
            os.makedirs(mats_dir)
        if not os.path.exists(feats_dir):
            os.makedirs(feats_dir)

        data_order = [self._data.get(i).get(self._data.get_id_key()) for i in range(self._data.get_size())]

        info = dict()
        info["data_dir"] = self._data.get_directory()
        info["id_key"] = self._data.get_id_key()
        info["size"] = len(self._dfmats)
        info["data_order"] = data_order
        info["lengths"] = list(self._lengths)

        with open(info_path, 'w') as fp:
            json.dump(info, fp)

        np.save(mask_path, self._mask)

        self._feature_seq_set.save(feats_dir)
        for i in range(len(self._dfmats)):
            np.save(join(mats_dir, str(i)), self._dfmats[i].get_matrix())

    @staticmethod
    def load(dir_path, data=None, data_cls=None):
        info_path = join(dir_path, "info")
        mats_path = join(dir_path, "mats")
        feats_dir = join(dir_path, "feats")
        mask_path = join(dir_path, "mask.npy")

        feature_seq_set = FeatureSequenceSet.load(feats_dir)

        obj = None
        with open(info_path, 'r') as fp:
            obj = json.load(fp)

        mats = []
        for i in range(obj["size"]):
            mats.append(np.load(join(mats_path, str(i) + ".npy")))

        if data is None:
            if data_cls is None:
                data_cls = DataSet
            data = data_cls.load(obj["data_dir"], id_key=obj["id_key"], order=obj["data_order"])
            return DataFeatureMatrixSequence(data, feature_seq_set, mats=mats, lengths=np.array(obj["lengths"]), mask=np.load(mask_path))
        else:
            mat_id_to_index = dict()
            for i in range(len(obj["data_order"])):
                mat_id_to_index[obj["data_order"][i]] = i
            # Perm : target index (data) -> source index (original mat)
            perm = [mat_id_to_index[data.get(i).get_id()] for i in range(data.get_size())]
            dfmats = DataFeatureMatrixSequence(data, feature_seq_set, mats=mats, lengths=np.array(obj["lengths"]), mask=np.load(mask_path))
            dfmats.reorder(perm, preordered_data=data)
            return dfmats


class MultiviewDataSet:
    def __init__(self, data=None, dfmats=None, dfmatseqs=None, ordering_seq=None):
        self._data = data
        if dfmats is None:
            self._dfmats = dict()
        else:
            self._dfmats = dfmats

        if dfmatseqs is None:
            self._dfmatseqs = dict()
        else:
            self._dfmatseqs = dfmatseqs

        self._ordering_seq = ordering_seq

    def __getitem__(self, key):
        if key in self._dfmats:
            return self._dfmats[key]
        else:
            return self._dfmatseqs[key]

    def get_data(self):
        return self._data

    def get_dfmats(self):
        return dict(self._dfmats)

    def get_size(self):
        return self._data.get_size()

    def shuffle(self):
        perm = self._data.shuffle()

        for dfmat in self._dfmats.values():
            dfmat.reorder(perm, preordered_data=self._data)

        for dfmatseq in self._dfmatseqs.values():
            dfmatseq.reorder(perm, preordered_data=self._data)

    def sort(self, key_fn):
        perm = self._data.sort(key_fn)

        for dfmat in self._dfmats.values():
            dfmat.reorder(perm, preordered_data=self._data)

        for dfmatseq in self._dfmatseqs.values():
            dfmatseq.reorder(perm, preordered_data=self._data)

    def partition(self, partition, key_fn):
        data_parts = self._data.partition(partition, key_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i

        mv_parts = dict()
        for key in data_parts.keys():
            mv_parts[key] = MultiviewDataSet()
            mv_parts[key]._data = data_parts[key]

        for name, dfmat in self._dfmats.items():
            dfmat_parts = dfmat._data_partition(data_parts, id_to_index, key_fn)
            for key, dfmat_part in dfmat_parts.items():
                mv_parts[key]._dfmats[name] = dfmat_part

        for name, dfmatseq in self._dfmatseqs.items():
            dfmatseq_parts = dfmatseq._data_partition(data_parts, id_to_index, key_fn)
            for key, dfmatseq_part in dfmatseq_parts.items():
                mv_parts[key]._dfmatseqs[name] = dfmatseq_part

        return mv_parts

    def union(self, other, first_views=None, second_views=None, missing_defaults=None):
        if first_views is None:
            first_views = set(self._dfmats.keys()).union(set(self._dfmatseqs.keys()))
        if second_views is None:
            second_views = first_views
        if missing_defaults is None:
            missing_defaults = dict()

        data_union = self._data.union(other)
        dfmats_union = dict()
        dfmatseqs_union = dict()

        all_views = first_views.union(second_views)
        for view in all_views:
            if view in first_views and view in second_views:
                if view in self._dfmats:
                    dfmats_union[view] = self._dfmats[view].union(other._dfmats[view], data=data_union)
                elif view in self._dfmatseqs:
                    dfmatseqs_union[view] = self._dfmatseqs[view].union(other._dfmatseqs[view], data=data_union)
                else:
                    raise ValueError("Missing view for union " + str(view))
            elif view in first_views and view not in second_views:
                if view not in missing_defaults:
                    raise ValueError("Ommitted view requires default: " + str(view))

                if view in self._dfmats:
                    default_mat = np.zeros(shape=other._dfmats[view].get_matrix().shape)
                    default_mat[:] = missing_defaults[view]
                    default_dfmat = DataFeatureMatrix(other._data, other._dfmats[view].get_feature_set(), init_features=False, mat=default_mat, compute_non_zero=False)
                    dfmats_union[view] = self._dfmats[view].union(default_dfmat, data=data_union)
                elif view in self._dfmatseqs:
                    raise NotImplementedError("Unioning sequence views not yet implemented.")
                else:
                    raise ValueError("Missing view for union " + str(view))
            elif view not in first_views and view in second_views:
                if view not in missing_defaults:
                    raise ValueError("Ommitted view requires default: " + str(view))

                if view in other._dfmats:
                    default_mat = np.zeros(shape=self._dfmats[view].get_matrix().shape)
                    default_mat[:] = missing_defaults[view]
                    default_dfmat = DataFeatureMatrix(self._data, self._dfmats[view].get_feature_set(), init_features=False, mat=default_mat, compute_non_zero=False)
                    dfmats_union[view] = default_dfmat.union(other._dfmats[view], data=data_union)
                elif view in self._dfmatseqs:
                    raise NotImplementedError("Unioning sequence views not yet implemented.")
                else:
                    raise ValueError("Missing view for union " + str(view))

        return MultiviewDataSet(data=data_union, dfmats=dfmats_union, dfmatseqs=dfmatseqs_union, ordering_seq=self._ordering_seq)

    def filter(self, filter_fn):
        data_filtered = self._data.filter(filter_fn)

        id_to_index = dict()
        for i in range(self._data.get_size()):
            id_to_index[self._data.get(i).get_id()] = i

        mv_filtered = MultiviewDataSet()
        mv_filtered._data = data_filtered

        for name, dfmat in self._dfmats.items():
            dfmat_filtered = dfmat._data_filter(data_filtered, id_to_index, filter_fn)
            mv_filtered._dfmats[name] = dfmat_filtered

        for name, dfmatseq in self._dfmatseqs.items():
            dfmatseq_filtered = dfmatseq._data_filter(data_filtered, id_to_index, filter_fn)
            mv_filtered._dfmatseqs[name] = dfmatseq_filtered

        return mv_filtered

    def get_random_subset(self, size):
        if size is None:
            return self
        if size > self._data.get_size():
            size = self._data.get_size()
        if size == 0:
            return self.filter(lambda d : False)
        subset_indices = np.random.choice(self.get_size(), size, replace=False)
        subset_ids = set([self._data.get(subset_indices[i]).get_id() for i in range(len(subset_indices))])
        filter_fn = lambda d : d.get_id() in subset_ids
        return self.filter(filter_fn)

    def get_subset(self, subset_i, size):
        if size is None:
            return self
        if size > self._data.get_size():
            size = self._data.get_size()
        if size == 0:
            return self.filter(lambda d : False)
        subset_indices = np.array(range(subset_i*size, (subset_i+1)*size))
        subset_ids = set([self._data.get(subset_indices[i]).get_id() for i in range(len(subset_indices))])
        filter_fn = lambda d : d.get_id() in subset_ids
        return self.filter(filter_fn)

    def get_random_batch(self, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        batch_indices = np.random.choice(self.get_size(), size, replace=False)
        return self.get_batch_by_indices(batch_indices, sort_lengths=sort_lengths,
                                         mat_views=mat_views, seq_views=seq_views, return_indices=return_indices)

    def get_batch(self, batch_i, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self.get_batch_by_indices(np.array(range(batch_i*size, (batch_i+1)*size)),
                                         sort_lengths=sort_lengths, mat_views=mat_views,
                                         seq_views=seq_views, return_indices=return_indices)

    def get_final_batch(self, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        final_size = self._data.get_size() % size
        if final_size == 0:
            return None
        return self.get_batch_by_indices(np.array(range(self._data.get_size() - final_size, self._data.get_size())),
                                         sort_lengths=sort_lengths, mat_views=mat_views,
                                         seq_views=seq_views, return_indices=return_indices)
    def get_num_batches(self, size):
        if size > self._data.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self._data.get_size() // size

    def get_batch_by_indices(self, batch_indices, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if mat_views is None:
            mat_views = self._dfmats.keys()
        if seq_views is None:
            seq_views = self._dfmatseqs.keys()
            if self._ordering_seq is None and len(seq_views) > 0:
                ordering_view = seq_views[0]
            else:
                ordering_view = self._ordering_seq

        # NOTE: Batches are sorted on the lengths of the first seq_view
        if sort_lengths and len(seq_views) > 0:
            lengths = self._dfmatseqs[ordering_view].get_lengths_by_indices(batch_indices)
            bls = [(batch_indices, lengths) for (batch_indices,lengths)
                    in sorted(zip(batch_indices, lengths), key=lambda p: -p[1])]
            batch_indices = np.array([bl[0] for bl in bls])

        batch_dict = dict()
        for mat_view in mat_views:
            batch_dict[mat_view] = self._dfmats[mat_view].get_batch_by_indices(batch_indices)

        for seq_view in seq_views:
            batch_dict[seq_view] = self._dfmatseqs[seq_view].get_batch_by_indices(batch_indices, sort_lengths=False)

        if return_indices:
            return batch_dict, batch_indices
        else:
            return batch_dict

    @staticmethod
    def load(data_path, dfmat_paths=dict(), dfmatseq_paths=dict(), data_cls=None, ordering_seq=None):
        mv = MultiviewDataSet(ordering_seq=ordering_seq)

        if data_cls is None:
            mv._data = DataSet.load(data_path)
        else:
            mv._data = data_cls.load(data_path)

        for name, path in dfmat_paths.items():
            mv._dfmats[name] = DataFeatureMatrix.load(path, data=mv._data)

        for name, path in dfmatseq_paths.items():
            mv._dfmatseqs[name] = DataFeatureMatrixSequence.load(path, data=mv._data)

        return mv

class PairedFeatureType:
    ZERO_ONE_DIFFERENCE = "ZERO_ONE_DIFFERENCE"
    SIGNED_DIFFERENCE = "SIGNED_DIFFERENCE"
    DIFFERENCE = "DIFFERENCE"
    TUPLE = "TUPLE"

class PairedMultiviewDataSet:
    def __init__(self, mvdata, data_indices, paired_feature_types):
        self._mvdata = mvdata
        self._data_indices = data_indices.astype(np.int32)
        self._paired_feature_types = paired_feature_types

    def get_dfmats(self):
        return self._mvdata.get_dfmats()

    def get_datum_pair(self, index):
        data = self._mvdata.get_data()
        return (data[self._data_indices[index,0]], data[self._data_indices[index,1]])

    def get_size(self):
        return self._data_indices.shape[0]

    def shuffle(self):
        perm = np.random.permutation(self.get_size())
        self._data_indices = self._data_indices[perm]

    def get_random_batch(self, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        batch_indices = np.random.choice(self.get_size(), size, replace=False)
        return self.get_batch_by_indices(batch_indices,
                                         mat_views=mat_views, return_indices=return_indices)

    def get_batch(self, batch_i, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self.get_batch_by_indices(np.array(range(batch_i*size, (batch_i+1)*size)),
                                         mat_views=mat_views,
                                         return_indices=return_indices)

    def get_final_batch(self, size, sort_lengths=True, mat_views=None, seq_views=None, return_indices=False):
        if size > self.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        final_size = self.get_size() % size
        if final_size == 0:
            return None
        return self.get_batch_by_indices(np.array(range(self.get_size() - final_size, self.get_size())),
                                        mat_views=mat_views,
                                        return_indices=return_indices)
    def get_num_batches(self, size):
        if size > self.get_size():
            raise ValueError("Batch size cannot be greater than data set size")
        return self.get_size() // size

    def get_batch_by_indices(self, batch_indices, mat_views=None, return_indices=False):
        mv_indices1 = self._data_indices[batch_indices,0]
        mv_indices2 = self._data_indices[batch_indices,1]
        
        batch_dict1 = self._mvdata.get_batch_by_indices(mv_indices1, mat_views=mat_views, seq_views=[])
        batch_dict2 = self._mvdata.get_batch_by_indices(mv_indices2, mat_views=mat_views, seq_views=[])

        if mat_views is None:
            mat_views = batch_dict1.keys()

        batch_dict = dict()
        for mat_view in mat_views:
            if isinstance(self._paired_feature_types[mat_view], dict):
                for paired_view, paired_view_type in self._paired_feature_types[mat_view].items():
                    batch_dict[paired_view] = self._compute_paired_value(batch_dict1[mat_view], batch_dict2[mat_view], paired_view_type)
            else:
                batch_dict[mat_view] = self._compute_paired_value(batch_dict1[mat_view], batch_dict2[mat_view], self._paired_feature_types[mat_view])

        if return_indices:
            return batch_dict, batch_indices
        else:
            return batch_dict

    def _compute_paired_value(self, mat1, mat2, paired_view_type):
        if paired_view_type == PairedFeatureType.DIFFERENCE:
            return mat1 - mat2
        elif paired_view_type == PairedFeatureType.SIGNED_DIFFERENCE:
            return self._untyped_sign(mat1 - mat2)
        elif paired_view_type == PairedFeatureType.ZERO_ONE_DIFFERENCE:
            return (self._untyped_sign(mat1 - mat2) + 1.0)/2.0
        elif paired_view_type == PairedFeatureType.TUPLE:
            return (mat1, mat2)
        else:
            raise ValueError(paired_view + " has invalid paired feature type.")

    def _untyped_sign(self, v):
        if isinstance(v, np.ndarray):
            return (v > 0).astype(np.float) - (v < 0).astype(np.float)
        else:
            return (v > 0).float() - (v < 0).float()

    def partition(self, partition, key_fn):
        data_indices_parts = dict()
        reverse_parts = partition.make_reverse_lookup()
        mv_parts = self._mvdata.partition(partition, key_fn)

        part_keys_to_indices = dict()
        for part_name, mv_part in mv_parts.items():
            part_keys_to_indices[part_name] = dict()
            for i in range(mv_part.get_size()):
                key = key_fn(mv_part.get_data().get(i))
                part_keys_to_indices[part_name][key] = i


        for i in range(self._data_indices.shape[0]):
            k1 = key_fn(self._mvdata.get_data().get(self._data_indices[i,0]))
            k2 = key_fn(self._mvdata.get_data().get(self._data_indices[i,1]))

            if k1 not in reverse_parts or k2 not in reverse_parts:
                continue

            if reverse_parts[k1] != reverse_parts[k2]:
                raise ValueError("Data pairs extend across partition parts (" + str(k1) + \
                ": " + str(reverse_parts[k1]) + ", " + str(k2) + ": " + str(reverse_parts[k2]) + ").")
            part_key = reverse_parts[k1]
            if part_key not in data_indices_parts:
                data_indices_parts[part_key] = []

            part_index_1 = part_keys_to_indices[part_key][k1]
            part_index_2 = part_keys_to_indices[part_key][k2]
            data_indices_parts[part_key].append((part_index_1, part_index_2))

        for part_key in partition.get_part_names():
            indices_array = None
            if part_key not in data_indices_parts:
                indices_array = np.array([])
            else:
                indices_array = np.zeros(shape=(len(data_indices_parts[part_key]),2))
                for i in range(len(data_indices_parts[part_key])):
                    indices_array[i,0] = data_indices_parts[part_key][i][0]
                    indices_array[i,1] = data_indices_parts[part_key][i][1]
            data_indices_parts[part_key] = indices_array

        pmv_parts = dict()
        for k, v in mv_parts.items():
            pmv_parts[k] = PairedMultiviewDataSet(v, data_indices_parts[k], self._paired_feature_types)

        return pmv_parts

    def union(self, other, first_views=None, second_views=None, missing_defaults=None):
        mvdata_union = self._mvdata.union(other._mvdata, first_views=first_views, second_views=second_views, missing_defaults=missing_defaults)
        data_indices_union = np.concatenate((self._data_indices, other._data_indices + self._mvdata.get_size()), axis=0)
        paired_feature_types_union = dict(self._paired_feature_types)

        return PairedMultiviewDataSet(mvdata_union, data_indices_union, paired_feature_types_union)

    def __getitem__(self, key):
        return self._mvdata[key] # NOTE: This may lead to problems...

    def get_data(self):
        raise NotImplementedError()

    def sort(self, key_fn):
        raise NotImplementedError()

    def filter(self, filter_fn):
        raise NotImplementedError()

    def get_random_subset(self, size):
        raise NotImplementedError()

    def get_subset(self, subset_i, size):
        raise NotImplementedError()

    def save(self, dir_path):
        info_path = join(dir_path, "info")
        indices_path = join(dir_path, "data_indices.npy")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        info = dict()
        info["paired_feature_types"] = self._paired_feature_types
        info["data_order"] = [self._mvdata.get_data().get(i).get_id() for i in range(self._mvdata.get_size())]
        with open(info_path, 'w') as fp:
            json.dump(info, fp)

        np.save(indices_path, self._data_indices)

    @classmethod
    def load(cls, dir_path, mvdata):
        info_path = join(dir_path, "info")
        indices_path = join(dir_path, "data_indices.npy")

        data_indices = np.load(indices_path)
        
        paired_feature_types = None
        data_order = None 
        with open(info_path, 'r') as fp:
            obj = json.load(fp)
            paired_feature_types = obj["paired_feature_types"]
            data_order = obj["data_order"]

        cur_data_to_indices = dict()
        for i in range(mvdata.get_size()):
            cur_data_to_indices[mvdata.get_data()[i].get_id()] = i 
        index_map = [cur_data_to_indices[data_order[i]] for i in range(len(data_order))]

        for i in range(data_indices.shape[0]):
            data_indices[i,0] = index_map[data_indices[i,0]]
            data_indices[i,1] = index_map[data_indices[i,1]]

        return PairedMultiviewDataSet(mvdata, data_indices, paired_feature_types)

    @classmethod
    def make(cls, mvdata, size, paired_feature_types, init_with_replacement=False, partition=None, partition_key_fn=None, filter_fn=lambda d1, d2 : True):
        data_indices = np.zeros(shape=(size,2), dtype=np.int32)
        
        reverse_partition = None
        if partition is not None:
            reverse_partition = partition.make_reverse_lookup()

        # TODO: Do this more efficiently without rejection sampling
        index_tuple_set = set()
        cur_size = 0
        while cur_size < data_indices.shape[0]:
            indices = np.random.choice(mvdata.get_size(), 2, replace=True)

            d0 = mvdata.get_data()[indices[0]]
            d1 = mvdata.get_data()[indices[1]]

            index_tuple = (indices[0], indices[1])
            if (not init_with_replacement and index_tuple in index_tuple_set) \
                or not filter_fn(d0, d1) \
                or (partition is not None and \
                    (partition_key_fn(d0) not in reverse_partition or partition_key_fn(d1) not in reverse_partition or reverse_partition[partition_key_fn(d0)] != reverse_partition[partition_key_fn(d1)])):
                continue
            index_tuple_set.add(index_tuple)

            data_indices[cur_size] = indices
            cur_size += 1

        return PairedMultiviewDataSet(mvdata, data_indices, paired_feature_types)

class MixedBatchData:
    def __init__(self, datas, proportions=None):
        self._datas = datas
        
        if proportions is None:
            self._proportions = np.zeros(shape=len(datas))
            self._proportions[:] = 1.0/len(datas)
        else:
            self._proportions = proportions

    def get_random_batch(self, size):
        data_index = np.random.choice(len(self._datas), size=1, p=self._proportions)[0]
        
        return self._datas[data_index].get_random_batch(size)

    def partition(self, partition, key_fn):
        data_parts = dict()
        for data in self._datas:
            parts = data.partition(partition, key_fn)
            for k, v in parts.items():
                if k not in data_parts:
                    data_parts[k] = []
                data_parts[k].append(v)

        mb_parts = dict()
        for part_name in partition.get_part_names():
            mb_parts[part_name] = MixedBatchData(data_parts[part_name], proportions=self._proportions)
        return mb_parts

register_feature_type(FeaturePathType)
register_feature_type(FeaturePathVectorDictionaryType)
register_feature_type(FeatureBucketedValueType)
register_feature_type(FeatureMatrixType)
register_feature_type(FeatureProductType)
register_feature_seq_type(FeatureMatrixSequence)
register_feature_seq_type(FeaturePathSequence)
