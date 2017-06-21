import numpy as np
import abc
from mung import data

# FIXME
class FeatureToken:
    def __init__(self):
        pass

class FeatureType(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_shape(self):
        """ Returns the shape of matrices computed by this feature """


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

    def equals(self, feature_type):
        if not isinstance(feature_type, FeatureMatrixType):
            return False
        if self._matrix_fn.__name__ != self._matrix_fn.__name__:  # FIXME Hack
            return False
        return True

    def init_start(self):
        pass

    def init_datum(self, datum):
        pass

    def init_end(self):
        pass

# END FIXME 

class FeatureSet:
    def __init__(self, feature_types=[]):
        self._feature_types = list(feature_types)
        self._size = sum([feature_type.get_size() for feature_type in feature_types])

    def has_feature_type(self, feature_type):
        for f in self._feature_types:
            if f.equals(feature_type):
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

    def compute(self, datum, start_from=0):
        v = []
        for i in range(start_from, len(self._feature_types)):
            v.extend(self._feature_types[i].compute(datum))
        return np.array(v)
    
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


class DataFeatureMatrix:
    def __init__(self, data, feature_set):
        self._data = data
        self._feature_set = feature_set
        self._mat = None
        self._nz_indices = None
        self._compute()

    def get_data(self):
        return self._data

    def get_feature_set(self):
        return self._feature_set

    def get_matrix(self):
        return self._mat

    def get_vector(self, i):
        return self._mat[i]

    def get_non_zero_indices(self, i):
        return self._nz_indices[i]

    def extend(self, feature_types):
        start_num = self._feature_set.get_num_feature_types()
        start_size = self._feature_set.get_size()
        added = self._feature_set.add_feature_types(feature_types)
        
        if len(added) == 0:
            return
       
        print "Extending feature matrix... (" + str(self._feature_set.get_size()-start_size) + " feature tokens)"
        #for added_f in added:
        #    for i in range(added_f.get_size()):
        #        print str(added_f.get_token(i))
        #print "\n"

        self._feature_set.init_start(start_from=start_num)
        for i in range(self._data.get_size()):
            self._feature_set.init_datum(self._data.get(i), start_from=start_num)
        self._feature_set.init_end(start_from=start_num)

        for i in range(self._data.get_size()):
        #    if i % 50 == 0:
        #        print "Extending on datum " + str(i)
            new_vec = self._feature_set.compute(self._data.get(i), start_from=start_num)

            nz = []
            for j in range(len(new_vec)):
                if new_vec[j] != 0:
                    nz.append(j + len(self._mat[i]))
            self._nz_indices[i].extend(nz)

            self._mat[i] = np.concatenate((self._mat[i], new_vec))

    def _compute(self):
        self._mat = []
        self._nz_indices = []
        
        self._feature_set.init_start()
        for i in range(self._data.get_size()):
            self._feature_set.init_datum(self._data.get(i))
        self._feature_set.init_end()

        #print "Computing feature matrix... (" + str(self._feature_set.get_size()) + " features)"
        for i in range(self._data.get_size()):
        #    if i % 50 == 0:
        #        print "Computing on datum " + str(i)
            v = self._feature_set.compute(self._data.get(i))
            self._mat.append(v)

            nz = []
            for j in range(len(v)):
                if v[j] != 0.0:
                    nz.append(j)
            self._nz_indices.append(nz)
        #print "Finished computing matrix"

    def shuffle(self):
        perm = np.random.permutation(len(self._mat))
        shuffled_mat = []
        shuffled_nz = []
        shuffled_data = []
        for i in range(len(perm)):
            shuffled_mat.append(self._mat[perm[i]])
            shuffled_nz.append(self._nz_indices[perm[i]])
            shuffled_data.append(self._data.get(perm[i]))
        self._data = data.DataSet(data=shuffled_data)
        self._mat = shuffled_mat
        self._nz_indices = shuffled_nz
