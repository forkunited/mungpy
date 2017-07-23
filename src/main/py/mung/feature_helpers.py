"""
Helper functions for constructing and saving feature matrices
"""

from mung.data import DataSet, Datum, Partition
from mung.feature import (FeaturePathType, FeaturePathSequence,
                         FeatureSet, FeatureSequenceSet,
                         DataFeatureMatrix, DataFeatureMatrixSequence,
                         ValueType)

def featurize_path_enum_seqs(input_data_dir, output_feature_dir, partition_file,
                             partition_fn, feature_name, paths, max_length,
                             init_data="train", token_fn=None, indices=False,
                             min_occur=2):
    """
    Constructs enumerable sequence features for a data set, and saves the
    resulting matrices and feature vocabulary to a directory

    Args:
        input_data_dir (:obj:`str`): Path to input data directory
        output_feature_dir (:obj:`str`): Path to output feature matrix directory
        partition_file (:obj:`str`): Path to file representing a partition of
            the data
        partition_fn (:obj:`data.Datum` -> :obj:`str`): Function to apply to
            data to construct keys on which to partition according to the
            partition loaded from partion_file
        feature_name (:obj:`str`): Name by which the feature will be referenced
        paths (:obj:`str list`): List of JSON paths into datums at which tokens
            of a sequence for the datum reside
        max_length (int): Maximum length of a sequence constructed for a datum.
            Sequences exceeding maximum length will be truncated

        init_data (:obj:`str`, optional): Name of the part of the partition to
            use to initialize the vocabulary for the feature (Defaults to
            "train")
        token_fn (:obj:`str` -> `:obj:`str`, optional): Function applied to
            each token of a sequence retrieved from a datum (Defaults to None)
        indices (bool, optional): Indicates whether the resulting feature
            sequences will be represented as lists of indices into a
            vocabulary, or lists of one-hot vectors representing those
            indices (Defaults to False)
        min_occur (int, optional): Minimum number of times a token must occur
            across all sequences in the initialization data for it to be
            included in the vocabulary.  Tokens that do not appear at least
            this number of times are replaced by `unc` symbols (Defaults to 2)
    """
    partition = Partition.load(partition_file)
    data_full = DataSet.load(input_data_dir)
    data_parts = data_full.partition(partition, partition_fn)

    value_type = ValueType.ENUMERABLE_ONE_HOT
    if indices:
        value_type = ValueType.ENUMERABLE_INDEX

    feat_seq = FeaturePathSequence(feature_name, paths, max_length,
                                   min_occur=min_occur, token_fn=token_fn,
                                   value_type=value_type)
    feat_seq_set = FeatureSequenceSet(feature_seqs=[feat_seq])
    feat_seq_set.init(data_parts[init_data])

    mat = DataFeatureMatrixSequence(data_full, feat_seq_set, init_features=False)
    mat.save(output_feature_dir)


def featurize_path_enum(input_data_dir, output_feature_dir, partition_file,
                        partition_fn, feature_name, paths, init_data="train",
                        token_fn=None, indices=False, min_occur=2):
    """
    Constructs enumerable features for a data set, and saves the
    resulting matrices and feature vocabulary to a directory

    Args:
        input_data_dir (:obj:`str`): Path to input data directory
        output_feature_dir (:obj:`str`): Path to output feature matrix directory
        partition_file (:obj:`str`): Path to file representing a partition of
            the data
        partition_fn (:obj:`data.Datum` -> :obj:`str`): Function to apply to
            data to construct keys on which to partition according to the
            partition loaded from partion_file
        feature_name (:obj:`str`): Name by which the feature will be referenced
        paths (:obj:`str list`): List of JSON paths into datums at which token
            values for the datum reside

        init_data (:obj:`str`, optional): Name of the part of the partition to
            use to initialize the vocabulary for the feature (Defaults to
            "train")
        token_fn (:obj:`str` -> `:obj:`str`, optional): Function applied to
            each token retrieved from a datum (Defaults to None)
        indices (bool, optional): Indicates whether the resulting feature
            sequences will be represented as indices into a vocabulary, or one-hot
            vectors representing those indices (Defaults to False)
        min_occur (int, optional): Minimum number of times a token must occur
            across the initialization data for it to be included in the
            vocabulary. Tokens that do not appear at least this number of times
            are replaced by `unc` symbols (Defaults to 2)
    """

    partition = Partition.load(partition_file)
    data_full = DataSet.load(input_data_dir)
    data_parts = data_full.partition(partition, partition_fn)

    value_type = ValueType.ENUMERABLE_ONE_HOT
    if indices:
        value_type = ValueType.ENUMERABLE_INDEX

    feat = FeaturePathType(feature_name, paths, min_occur=min_occur,
                           value_type=value_type)
    feat_set = FeatureSet(feature_types=[feat])
    feat_set.init(data_parts[init_data])

    mat = DataFeatureMatrix(data_full, feat_set, init_features=False)
    mat.save(output_feature_dir)


def featurize_path_scalars(input_data_dir, output_feature_dir, partition_file,
                           partition_fn, feature_name, paths, init_data="train"):
    """
    Constructs scalar features for a data set, and saves the resulting matrices
    and feature vocabulary to a directory

    Args:
        input_data_dir (:obj:`str`): Path to input data directory
        output_feature_dir (:obj:`str`): Path to output feature matrix directory
        partition_file (:obj:`str`): Path to file representing a partition of
            the data
        partition_fn (:obj:`data.Datum` -> :obj:`str`): Function to apply to
            data to construct keys on which to partition according to the
            partition loaded from partion_file
        feature_name (:obj:`str`): Name by which the feature will be referenced
        paths (:obj:`str list`): List of JSON paths into datums at which scalar
            values or lists of scalar values reside

        init_data (:obj:`str`, optional): Name of the part of the partition to
            use to initialize the vocabulary for the feature (Defaults to
            "train")
    """

    partition = Partition.load(partition_file)
    data_full = DataSet.load(input_data_dir)
    data_parts = data_full.partition(partition, partition_fn)

    feat = FeaturePathType(feature_name, paths,
                           value_type=ValueType.SCALAR)
    feat_set = FeatureSet(feature_types=[feat])
    feat_set.init(data_parts[init_data])

    mat = DataFeatureMatrix(data_full, feat_set, init_features=False)
    mat.save(output_feature_dir)
