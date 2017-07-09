from mung.data import DataSet, Datum, Partition
from mung.feature import FeaturePathSequence, FeatureSequenceSet, DataFeatureMatrixSequence, ValueType

def featurize_path_enum_seqs(input_data_dir, output_feature_dir, partition_file,
                             partition_fn, feature_name, paths, max_length,
                             init_data="train", token_fn=None):
    partition = Partition.load(partition_file)
    data_full = DataSet.load(input_data_dir)
    data_parts = data_full.partition(partition, partition_fn)

    feat_seq = FeaturePathSequence(feature_name, paths, max_length, min_occur=2,
                                   token_fn=token_fn)
    feat_seq_set = FeatureSequenceSet(feature_seqs=[feat_seq])
    feat_seq_set.init(data_parts[init_data])

    mat = DataFeatureMatrixSequence(data_full, feat_seq_set, init_features=False)
    mat.save(output_feature_dir)


def featurize_path_scalars(input_data_dir, output_feature_dir, partition_file,
                           partition_fn, feature_name, paths, init_data="train"):
    partition = Partition.load(partition_file)
    data_full = DataSet.load(input_data_dir)
    data_parts = data_full.partition(partition, partition_fn)

    feat = FeaturePathType(feature_name, paths,
                           value_type=ValueType.SCALAR)
    feat_set = FeatureSet(feature_types=[feat])
    feat_set.init(data_parts[init_data])

    mat = DataFeatureMatrix(data_full, feat_set, init_features=False)
    mat.save(output_feature_dir)
