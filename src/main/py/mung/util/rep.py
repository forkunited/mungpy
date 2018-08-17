import csv
import numpy as np
import codecs

class StorageFormat:
    VEC = "VEC"

class StoredVectorDictionary:
    def __init__(self, vecs=None, file_path=None, 
                 storage_format=StorageFormat.VEC, delimiter=" ", 
                 header_lines=1, transform_fn=None, normalized=False):
        if vecs is not None:
            self._vecs = vecs
        else:
            self._vecs = dict()
            if file_path is not None:
                self._load(file_path, storage_format, delimiter, header_lines, transform_fn, normalized)
        if len(self._vecs) == 0:
            self._vec_size = 0
        else:
            self._vec_size = self._vecs.values()[0].shape[0]

    def to_dict(self):
        return self._vecs

    def get_default_vector(self):
        return np.zeros(shape=(self._vec_size))

    def __getitem__(self, key):
        return self._vecs[key]
        
    def __contains__(self, key):
        return key in self._vecs 

    def __len__(self):
        return len(self._vecs)

    def _load(self, file_path, storage_format, delimiter, header_lines, transform_fn, normalized):
        if storage_format == StorageFormat.VEC:
            self._load_from_vec(file_path, delimiter, header_lines, transform_fn, normalized)
        else:
            raise ValueError("Invalid storage format for vector dictionary: " + str(storage_format))
    
    def _load_from_vec(self, file_path, delimiter, header_lines, transform_fn, normalized):
        self._vecs = dict()
        lines = None
        with codecs.open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
        vec_size = 0
        for i in range(header_lines, len(lines)):
            line_parts = lines[i].split(delimiter)
            key = line_parts[0]
            vec = np.array([float(val) for val in line_parts[1:]])
            if transform_fn is not None:
                vec = transform_fn(vec)
            self._vecs[key] = vec
            vec_size = max(vec_size, self._vecs[key].shape[0])

        if normalized:
            mat = np.zeros(shape=(len(self._vecs), vec_size))
            i = 0
            for key in self._vecs.keys():
                mat[i] = self._vecs[key] 
                i += 1
            mu = np.mean(mat, axis=0)
            sigma = np.std(mat, axis=0)

            for key in self._vecs.keys():
                self._vecs[key] = (self._vecs[key] - mu)/sigma