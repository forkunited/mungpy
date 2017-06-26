import numpy as np
import abc
from mung import data

class FeatureToken(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        """ Returns a string representation of the feature """

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
    def get_size(self):
        """ Returns the size of the vectors computed by this feature """

    @abc.abstractmethod
    def compute(self, vec, start_index):
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

class FeatureSequence(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, feature_seq):
        """ Returns whether two feature sequences are the same """

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


class FeatureMatrixToken(FeatureToken):
    def __init__(self, name, index):
        FeatureToken.__init__(self)
        self._name = name
        self._index = index

    def __str__(self):
        return self._name + "_" + str(self._index)

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

    def compute(self, datum):
        return self._matrix_fn(datum).flatten().tolist()

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


class DataFeatureMatrix:
    def __init__(self, data, feature_set, init_features=True):
        self._data = data
        self._feature_set = feature_set
        self._mat = None
        self._nz_indices = None
        self._compute(init_features=init_features)

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


class DataFeatureMatrixSequence:
    def __init__(self, data, feature_seq_set):
        self._data = data
        self._feature_seq_set = feature_seq_set
        self._dfmats = []
        self._compute()

    def get_data(self):
        return self._data

    def get_feature_seq_set(self):
        return self._feature_seq_set

    def get_matrix(self, seq_i):
        return # FIXME

    def get_vector(self, seq_i, data_i):
        return # FIXME

    def get_batch(self, seq_i, batch_i, size):
        return # FIXME

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

