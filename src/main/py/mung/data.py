import numpy as np

class Datum:
    # FIXME Later for annotators init_datum=None, properties=None
    def __init__(self, datum_id=None, datum_type=None)
        # FIXME If id is None, generate one
        self._datum_id = datum_id
        self._datum_type = datum_type
        self._properties = dict()

    def get_id(self):
        return self._datum_id

    def get_type(self):
        return self._datum_type

    def has_property(self, name):
        return (name in self._properties)

    def get_property(self, name):
        # FIXME: This should handle reference datums 
        # at some point
        return self._properties[name]
    
    def copy(self, deep=False):
        # FIXME Do this later
        pass

    def __eq__(self, other):
        if isinstance(other, Datum):
            return self._datum_id == other._datum_id
        return NotImplemented

    def __hash__(self):
        return hash(self._datum_id)


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

