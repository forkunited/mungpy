import numpy as np
import abc
import json
import dill as pickle
from mung import data
from bidict import bidict
from collections import Counter

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
    def save(self, file_path):
        """ Save a representation of the feature to file """

    @staticmethod
    def load(data_dir, id_key="id"):


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
        """ Returns the length of the feature sequence """

    @abc.abstractmethod
    def get_type(self, index)
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
    def __init__(self, name, matrix_fn, size):
        FeatureType.__init__(self)
        self._name = name
        self._matrix_fn = matrix_fn
        self._size = size

    def compute(self, datum, vec. start_index):
        vec[start_index:start_index+self._size] = self._matrix_fn(datum).flatten()
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
        if self._name != feature_type.name:
            return False
        return True

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeatureMatrixType"
        obj["name"] = self._name
        obj["matrix_fn"] = pickle.dumps(self._matrix_fn)
        obj["size"] = self._size
        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            return FeatureMatrixType.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        matrix_fn = pickle.loads(obj["matrix_fn"])
        size = obj["size"]
        return FeatureMatrixType(name, matrix_fn, size)


class FeatureMatrixSequence(FeatureSequence):
    def __init__(self, name, matrix_fn, sequence_length, feature_size):
        FeatureSequence.__init__(self)
        self._name = name
        self._matrix_fn = matrix_fn
        self._sequence_length = sequence_length
        self._feature_size = feature_size

        self._types = []
        for i in range(self._sequence_length):
            def vector_fn(datum):
                mat = matrix_fn(datum)
                if i >= mat.shape()[0]:
                    return np.zeros(self._feature_size)
                else:
                    return mat[i]

            self._types.append(FeatureMatrixType(self, self._name + "." + str(i), vector_fn, self._feature_size))

    def __eq__(self, feature_seq):
        if not isinstance(feature_seq, FeatureMatrixSequence):
            return False
        if self._name != feature_seq.name:
            return False
        return True

    def get_name(self):
        return self._name

    def get_size(self):
        return self._sequence_length

    def get_type(self, index)
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
        obj["sequence_length"] = self._sequence_length
        obj["feature_size"] = self._feature_size
        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
            return FeatureMatrixSequence.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        name = obj["name"]
        matrix_fn = pickle.loads(obj["matrix_fn"])
        sequence_length = obj["sequence_length"]
        feature_size = obj["feature_size"]
        return FeatureMatrixSequence(name, matrix_fn, sequence_length, feature_size)


class FeaturePathToken(FeatureToken):
    def __init__(self, name, index, token):
        FeatureToken.__init__(self)
        self._name = name
        self._index = index
        self._token = token

    def __str__(self):
        return self._name + "_" + str(self._token)

    def get_name(self):
        return self._name

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

class FeaturePathType(FeatureType):
    VALUE_ENUMERABLE_ONE_HOT = 0
    VALUE_SCALAR = 1

    SYMBOL_SEQ_START = "SYM_START"
    SYMBOL_SEQ_MID = "SYM_MID"
    SYMBOL_SEQ_END = "SYM_END"
    SYMBOL_SEQ_UNC = "SYM_UNC"

    def __init__(self, name, paths, min_occur=2, no_init=False, value_type=FeaturePathType.VALUE_ENUMERABLE_ONE_HOT, value_fn=None, seq_index=None, vocab=None):
        FeatureType.__init__(self)
        self._name = name
        self._paths = paths
        self._min_occur = min_occur
        self._no_init = no_init
        self._value_type = value_type
        self._value_fn = value_fn
        self._seq_index = seq_index

        self._counter = None
        self._vocab = vocab

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
                continue
            if isinstance(values[0], list):
                if self._value_type != FeaturePathType.VALUE_SCALAR:
                    new_values = []
                    new_values.append(FeaturePathType.SYMBOL_SEQ_START)
                    for value in values:
                        new_values.extend(value)
                       new_values.append(FeaturePathType.SYMBOL_SEQ_MID)
                    new_values[len(new_values)-1] = FeaturePathType.SYMBOL_SEQ_END
                    values = new_values[:seq_index+1]
                else:
                    values = [el for el in value for value in values][:seq_index+1]

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
                values = [el for el in value for value in values]
            for i in range(len(values)):
                mapping.append((path + "_" + str(i), values[i]))

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
            if self._value_type == FeaturePathType.VALUE_SCALAR:
                index = self._vocab[path_value[0]]
                value = path_value[1]
            else:
                index = self._vocab[path_value[0] + "_" + path_value[1]]
                value = 1.0
            vec[start_index + index] = value

    def get_size(self):
        return len(self._vocab)

    def get_token(self, index):
        return FeaturePathToken(self._name, index, self._vocab.inv[index])

    def __eq__(self, feature_type):
        if not isinstance(feature_type, FeaturePathType):
            return False
        if self._name != feature_type.name:
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
            if self._value_type == FeaturePathType.VALUE_SCALAR:
                for path_values in path_values_seq:
                    for path_value in path_values:
                        self._counter[path_value[0]] += 1
            else:
                for path_values in path_values_seq:
                    for path_value in path_values:
                        self._counter[path_value[0] + "_" + path_value[1]] += 1
        else:
            path_values = self._get_path_values(datum)
            if self._value_type == FeaturePathType.VALUE_SCALAR:
                for path_value in path_values:
                    self._counter[path_value[0]] += 1
            else:
                for path_value in path_values:
                    self._counter[path_value[0] + "_" + path_value[1]] += 1

    def init_end(self):
        if self._no_init:
            return
        vocab_list = []
        for key, count in self._counter:
            if self._value_type == FeaturePathType.VALUE_SCALAR or count >= self._min_occur:
                vocab_list.append(key)
        vocab_list.sort()

        self._vocab = bidict()
        if self._seq_index is not None and self._value_type != FeaturePathType.VALUE_SCALAR:
            self._vocab[FeaturePathType.SYMBOL_SEQ_UNC] = 0
            self._vocab[FeaturePathType.SYMBOL_SEQ_START] = 1
            self._vocab[FeaturePathType.SYMBOL_SEQ_END] = 2
            self._vocab[FeaturePathType.SYMBOL_SEQ_MID] = 3
            
            index = 4
            for v in vocab_list:
                self._vocab[v] = index
                index += 1
        else:
            index = 0
            for v in vocab_list:
                self._vocab[v] = index
                index += 1
        
        self._counter = None

    def save(self, file_path):
        obj = dict()
        obj["type"] = "FeaturePathType"
        obj["name"] = self._name
        obj["paths"] = self._paths
        obj["min_occur"] = self._min_occur
        obj["no_init"] = self._no_init
        obj["value_type"] = self._value_type
        if self._value_fn is not None:
            obj["value_fn"] = pickle.dumps(self._value_fn)
        if self._seq_index is not None:
            obj["seq_index"] = self._seq_index
        if self._vocab is not None:
            obj["vocab"] = dict(self._vocab)
        
        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
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
            value_fn = pickle.loads(self._value_fn)
        seq_index = None
        if "seq_index" in obj:
            seq_index = obj["seq_index"]
        vocab = None
        if "vocab" in obj:
            vocab = bidict(obj["vocab"]) 

        return FeaturePathType(name, paths, min_occur=min_occur, no_init=no_init, value_type=value_type, value_fn=value_fn, seq_index=seq_index, vocab=vocab)


class FeaturePathSequence(FeatureSequence):
    def __init__(self, name, paths, seq_length, min_occur=2, value_type=FeaturePathType.VALUE_ENUMERABLE_ONE_HOT, value_fn=None, vocab=None):
        FeatureSequence.__init__(self)
        self._name = name
        self._paths = paths
        self._seq_length = sequence_length
        self._min_occur = min_occur
        self._value_type = value_type
        self._value_fn = value_fn

        self._vocab = None
        self._init_type = None
        self._types = []

        if vocab is not None:
            self._vocab = vocab
            self._init_types()

    def _init_types(self):
        self._types = []
        for i in range(self._seq_length):
            self._types.append(FeaturePathType(name,
                self._paths,
                min_occur=self._min_occur,
                no_init=True,
                value_type=self._value_type,
                value_fn=self._value_fn,
                seq_index=i,
                vocab=self._vocab
            ))

    def __eq__(self, feature_seq):
        if not isinstance(feature_seq, FeaturePathSequence):
            return False
        if self._name != feature_seq.name:
            return False
        return True

    def get_name(self):
        return self._name

    def get_size(self):
        return self._seq_length

    def get_type(self, index)
        return self._types[index]

    def init_start(self):
        self._vocab = bidict()
        self._init_type = FeaturePathType(
            name, 
            self._paths, 
            min_occur=self._min_occur, 
            no_init=False, 
            value_type=self._value_type, 
            value_fn=self._value_fn, 
            seq_index=self._seq_length-1, 
            vocab=self._vocab)
        self._init_type.init_start()

    def init_datum(self, datum):
        self._init_type.init_datum(datum)

    def init_end(self):
        self._init_type.init_end()
        self._types = self._init_types()
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
        if self._vocab is not None:
            obj["vocab"] = dict(self._vocab)

        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fp:
            obj = json.load(fp)
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
            value_fn = pickle.loads(self._value_fn)
        vocab = None
        if "vocab" in obj:
            vocab = bidict(obj["vocab"])
        return FeaturePathSequence(name, paths, seq_length, min_occur=min_occur, value_type=value_type, value_fn=value_fn, vocab=vocab)


class FeatureSet:
    def __init__(self, feature_types=[]):
        self._feature_types = list(feature_types)
        self._size = sum([feature_type.get_size() for feature_type in feature_types])

    def has_feature_type(self, feature_type):
        for f in self._feature_types:
            if f == feature_type:
                return True
        return False

    def add_feature_type(self, feature_type):
        if self.has_feature_type(feature_type):
            return False
        self._feature_types.append(feature_type)
        self._size += feature_type.get_size()
        return True

    def add_feature_types(self, feature_types):
        ret = []
        for feature_type in feature_types:
            r = self.add_feature_type(feature_type)
            if r:
                ret.append(feature_type)
        return ret

    def get_size(self):
        return self._size

    def get_num_feature_types(self):
        return len(self._feature_types)

    def compute(self, datum, start_from=0, v=None):
        if v is None:
            v = np.array(self.get_size())

        for i in range(0, len(self._feature_types)):
            if i >= start_from:
                self._feature_types[i].compute(datum, v, start_index)
            start_index += self._feature_types[i].get_size()
        return v
    
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
            if index < offset + self._feature_types[i].get_size():
                return self._feature_types[i].get_token(index - offset)
            else:
                offset += self._feature_types[i].get_size()
        return None

    def get_feature_type(self, index):
        return self._feature_types[index]

    def copy(self):
        return FeatureSet(feature_types=self._feature_types)

    def save(self, dir_path):
        for feature_type in self._feature_types:
            feature_type.save(join(dir_path, feature_type.get_name()))           

    @staticmethod
    def load(dir_path):
        file_paths = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]       
        feature_types = []
        for file_path in file_paths:
            with open(file_path, 'r') as fp:
                obj = json.load(fp)
                if obj["type"] == "FeaturePathType":
                    feature_types.append(FeaturePathType.from_dict(obj))
                elif obj["type"] == "FeatureMatrixType":
                    feature_types.append(FeatureMatrixType.from_dict(obj)) 
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
        for i in range(self._feature_seqs[0].get_size()):
            feature_set = FeatureSet(feature_types=[feature_seq.get_type(i) for feature_seq in self._feature_seqs])
            self._feature_sets.append(feature_set)

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

    def init_start(self, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_start()

    def init_datum(self, datum, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_datum(datum)

    def init_end(self, start_from=0):
        for i in range(start_from, len(self._feature_seqs)):
            self._feature_seqs[i].init_end()

    def get_feature_token(self, index, seq_index=0):
        if len(self._feature_sets) == 0:
            return None
        return self._feature_sets[seq_index].get_feature_token(index)

    def get_feature_seq(self, index):
        return self._feature_seqs[index]

    def get_feature_set(self, index):
        return sefl._feature_sets[index]

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
            with open(file_path, 'r') as fp:
                obj = json.load(fp)
                if obj["type"] == "FeaturePathSequence":
                    feature_seqs.append(FeaturePathSequence.from_dict(obj))
                elif obj["type"] == "FeatureMatrixSequence":
                    feature_seqs.append(FeatureMatrixSequence.from_dict(obj))
        return FeatureSequenceSet(feature_seqs=feature_seqs)


class DataFeatureMatrix:
    def __init__(self, data, feature_set, init_features=True, mat=None):
        self._data = data
        self._feature_set = feature_set
        self._mat = None
        self._nz_indices = None
        if mat is None:
            self._compute(init_features=init_features)
        else:
            self._mat = mat
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

    def get_matrix(self):
        return self._mat

    def get_vector(self, i):
        return self._mat[i]

    def get_batch(self, i, size):
        return self._mat[i*size:(i+1)*size]

    def get_num_batches(self, size):
        return self._data.get_size() // size

    def get_non_zero_indices(self, i):
        return self._nz_indices[i]

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

            nz = []
            for j in range(start_size, len(self._mat[i])):
                if self._mat[i,j] != 0:
                    nz.append(j)
            self._nz_indices[i].extend(nz)

    def _compute(self, init_features=True):
        self._mat = np.zeros((data.get_size(), feature_set.get_size()))
        self._nz_indices = []
        
        if init_features:
            self._feature_set.init_start()
            for i in range(self._data.get_size()):
                self._feature_set.init_datum(self._data.get(i))
            self._feature_set.init_end()

        #print "Computing feature matrix... (" + str(self._feature_set.get_size()) + " features)"
        for i in range(self._data.get_size()):
            self._feature_set.compute(self._data.get(i), v=self._mat[i])

            nz = []
            for j in range(feature_set.get_size()):
                if self._mat[i][j] != 0.0:
                    nz.append(j)
            self._nz_indices.append(nz)
        #print "Finished computing matrix"

    def shuffle(self):
        perm = np.random.permutation(len(self._mat))
        self.reorder(perm)

    def reorder(self, perm, preordered_data=None):
        shuffled_mat = np.zeros(self._mat.shape())
        shuffled_nz = []
        shuffled_data = []
        for i in range(len(perm)):
            np.copyto(shuffled_mat[i], self._mat[perm[i]]) 
            shuffled_nz.append(self._nz_indices[perm[i]])
            shuffled_data.append(self._data.get(perm[i]))

        if preordered_data is None:
            self._data = data.DataSet(data=shuffled_data)
        else:
            self._data = preordered_data

        self._mat = shuffled_mat
        self._nz_indices = shuffled_nz

    def save(self, dir_path):
        info_path = join(dir_path, "info")
        mat_path = join(dir_path, "mat")
        feats_dir = join(dir_path, "feats")
        
        info = dict()
        info["data_dir"] = self._data.get_directory()
        info["id_key"] = self._data.get_id_key()        
        with open(info_path, 'w') as fp:
            json.dump(info, fp)
         
        np.save(mat_path, self._mat)
        self._feature_set.save(feats_dir)

    @staticmethod
    def load(dir_path):
        info_path = join(dir_path, "info")
        mat_path = join(dir_path, "mat")
        feats_dir = join(dir_path, "feats")

        data = None
        with open(info_path, 'r') as fp:
            obj = json.load(fp)
            data = DataSet.load(obj["data_dir"], id_key=obj["id_key"])

        mat = np.load(mat_path)
        feature_set = FeatureSet.load(feats_dir)

        return DataFeatureMatrix(data, feature_set, init_features=False, mat=mat)


class DataFeatureMatrixSequence:
    def __init__(self, data, feature_seq_set, mats=None):
        self._data = data
        self._feature_seq_set = feature_seq_set
        self._dfmats = []
        if dfmats is None:
            self._compute()
        else:
            for mat in mats:
                self._dfmats.append(DataFeatureMatrix(data, self._feature_seq_set.get_feature_set(i), init_features=False, mat=mat))

    def get_data(self):
        return self._data

    def get_feature_seq_set(self):
        return self._feature_seq_set

    def get_matrix(self, seq_i):
        return self._dfmats[seq_i]

    def get_vector(self, seq_i, data_i):
        return self._dfmats[seq_i].get_vector(data_i)

    def get_batch(self, seq_i, batch_i, size):
        return self._dfmats[seq_i].get_batch(batch_i, size)

    def get_num_batches(self, size):
        return self._data.get_size() // size

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

        for i in range(self._feature_seq_set.get_size()):
            feature_set = self._feature_seq_seq.get_feature_set(i)
            feature_set.extend(feature_seqs, start_num=start_num, start_size=start_size)

    def _compute(self):
        self._feature_seq_set.init_start()
        for i in range(self._data.get_size()):
            self._feature_seq_set.init_datum(self._data.get(i))
        self._feature_seq_set.init_end()

        for i in range(self._feature_seq_set.get_size()):
            feature_set = self._feature_seq_set.get_feature_set(i)
            self._dfmats.append(DataFeatureMatrix(self._data, feature_set, init_features=False))

    def shuffle(self):
        perm = np.random.permutation(self._data.get_size())
        self.reorder(perm)

    def reorder(self, perm):
        shuffled_data = []
        for i in range(len(perm)):
            shuffled_data.append(self._data.get(perm[i]))
        self._data = shuffled_data

        for dfmat in self._dfmats:
            dfmat.reorder(perm, preordered_data=self._data)

    def save(self, dir_path):
        info_path = join(dir_path, "info")
        mats_dir = join(dir_path, "mats")
        feats_dir = join(dir_path, "feats")

        info = dict()
        info["data_dir"] = self._data.get_directory()
        info["id_key"] = self._data.get_id_key()
        info["size"] = len(self._dfmats)
        with open(info_path, 'w') as fp:
            json.dump(info, fp)

        self._feature_seq_set.save(feats_dir)
        for i in range(len(self._dfmats)):
            np.save(join(mats_dir, str(i)), dfmat.get_matrix())

    @staticmethod
    def load(dir_path):
        info_path = join(dir_path, "info")
        mats_path = join(dir_path, "mats")
        feats_dir = join(dir_path, "feats")

        data = None
        with open(info_path, 'r') as fp:
            obj = json.load(fp)
            data = DataSet.load(obj["data_dir"], id_key=obj["id_key"])

        feature_set = FeatureSequenceSet.load(feats_dir)

        mats = []
        for i in range(obj["size"]):
            mats.append(np.load(join(mats_path, str(i))))

        return DataFeatureMatrixSequence(data, feature_seq_set, mats=mats)
