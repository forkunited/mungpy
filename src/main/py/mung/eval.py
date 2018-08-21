import abc
import numpy as np
import sklearn.metrics
import scipy.stats

class Evaluation(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, data_parameters, trials=1):
        super(Evaluation, self).__init__()
        self._data = data
        self._data_parameters = data_parameters
        self._name = name
        self._trials = trials

    @abc.abstractmethod
    def run(self, model):
        """ Runs evaluation on a model """

    def set_data(self, data):
        self._data = data

    def get_trials(self):
        return self._trials

    def get_name(self):
        return self._name

    @staticmethod
    def run_all(evaluations, model, flatten_result=True):
        results = dict()
        for evaluation in evaluations:
            result = evaluation.run(model)
            if isinstance(result, dict) and flatten_result:
                for key, value in result.iteritems():
                    results[evaluation.get_name() + "_" + key] = value
            elif isinstance(result, tuple) and flatten_result:
                for i in range(len(result)):
                    results[evaluation.get_name() + "_" + str(i)] = result[i]
            else:
                results[evaluation.get_name()] = result

        return results

class PredictionMetric(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        super(PredictionMetric, self).__init__()
        self._name = name

    def get_name(self):
        return self._name

    @abc.abstractmethod
    def compute(self, y_true, y_pred):
        """ Return value based on predictions and ground truth """

class ClassificationReport(PredictionMetric):
    def __init__(self, name, labels=None, 
            target_names=None, sample_weight=None, digits=2):
        super(ClassificationReport, self).__init__(name)
        self._labels = labels
        self._target_names = target_names
        self._sample_weight = sample_weight
        self._digits = digits

    def compute(self, y_true, y_pred):
        return sklearn.metrics.classification_report(y_true.astype(int), y_pred.astype(int), 
            self._labels, self._target_names, self._sample_weight, self._digits)

class ConfusionMatrix(PredictionMetric):
    def __init__(self, name, labels=None, sample_weight=None, target_names=None):
        super(ConfusionMatrix, self).__init__(name)
        self._labels = labels
        self._sample_weight = sample_weight
        self._target_names = target_names

    def compute(self, y_true, y_pred):
        conf = sklearn.metrics.confusion_matrix(y_true.astype(int), y_pred.astype(int), 
            labels=self._labels, sample_weight=self._sample_weight)
        return Table(conf, row_labels=self._target_names, column_labels=self._target_names)

class F1(PredictionMetric):
    def __init__(self, name, labels=None, pos_label=1, average='weighted', sample_weight=None):
        super(F1, self).__init__(name)
        self._labels = labels
        self._pos_label = pos_label
        self._average = average
        self._sample_weight = sample_weight

    def compute(self, y_true, y_pred):
        return sklearn.metrics.f1_score(y_true.astype(int), y_pred.astype(int),
            labels=self._labels, pos_label=self._pos_label, average=self._average, 
            sample_weight=self._sample_weight)

class SpearmanR(PredictionMetric):
    def __init__(self, name, include_p=True):
        super(SpearmanR, self).__init__(name)
        self._include_p = include_p

    def compute(self, y_true, y_pred):
        if self._include_p:
            return scipy.stats.spearmanr(y_true, y_pred)
        else:
            return scipy.stats.spearmanr(y_true, y_pred)[0]

class PearsonR(PredictionMetric):
    def __init__(self, name, include_p=True):
        super(PearsonR, self).__init__(name)
        self._include_p = include_p

    def compute(self, y_true, y_pred):
        if self._include_p:
            return scipy.stats.pearsonr(y_true, y_pred)
        else:
            return scipy.stats.pearsonr(y_true, y_pred)[0]

class MAE(PredictionMetric):
    def __init__(self, name, sample_weight=None, multioutput='uniform_average'):
        super(MAE, self).__init__(name)
        self._sample_weight = sample_weight
        self._multioutput = multioutput
    
    def compute(self, y_true, y_pred):
        return sklearn.metrics.mean_absolute_error(y_true, y_pred, \
            sample_weight=self._sample_weight, multioutput=self._multioutput)

class Accuracy(PredictionMetric):
    def __init__(self, name, tolerance=0.0):
        super(Accuracy, self).__init__(name)
        self._tolerance = tolerance
    
    def compute(self, y_true, y_pred):
        return np.sum((np.abs(y_true - y_pred) <= self._tolerance).astype(np.float))/y_pred.shape[0]

class FoldedCV:
    # data : MultiviewDataSet or (key -> MultiviewDataSet)
    # partitions : Partition or (key -> Partition) (each partition should have the same set of part names)
    def __init__(self, data, partitions, load_model_fn, train_fn, load_evaluation_fn, 
                    separate_dev_test=True, exclude_parts=set()):
        if isinstance(data, dict):
            self._data = data
            self._partitions = partitions
        else:
            self._data = dict()
            self._partitions = dict()
            self._data[""] = data
            self._partitions[""] = partitions
        
        self._load_model_fn = load_model_fn
        self._train_fn = train_fn
        self._load_evaluation_fn = load_evaluation_fn
        self._separate_dev_test = separate_dev_test
        self._exclude_parts = exclude_parts

    def run(self, model_config, train_evaluation_config, \
            dev_evaluation_config, trainer_config, logger):
        part_names = list(set(self._partitions.itervalues().next().get_part_names()) - self._exclude_parts)
        k = len(part_names) # k-fold cv with parts
        results = []
        for i in range(k): # Iterate over folds
            partitions_i = self._make_fold_partition(part_names, i)
            datas_i = dict()
            D_params = None

            # Construct train/dev/test for current fold (i)
            for key, d in self._data.iteritems():
                partition = partitions_i[key]
                d_parts = d.partition(partition, lambda d : d.get_id())

                if not self._separate_dev_test:
                    d_parts["dev"] = d_parts["test"]
                
                suffix = ""
                if len(key) > 0:
                    suffix = "_" + key

                datas_i["train" + suffix] = d_parts["train"]
                datas_i["dev" + suffix] = d_parts["dev"]
                datas_i["test" + suffix] = d_parts["test"]

                D_params = d_parts["train"]

            # Load model and evaluations
            data_parameter, model = self._load_model_fn(model_config, D_params)
            train_evaluations = self._load_evaluation_fn(train_evaluation_config, datas_i, data_parameter)
            dev_evaluations = self._load_evaluation_fn(dev_evaluation_config, datas_i, data_parameter)

            # Train and evaluate on current fold (i)
            _, best_model, _ = self._train_fn(trainer_config, data_parameter, \
                model.get_loss_criterion(), logger, train_evaluations, model, datas_i)
            results.append(Evaluation.run_all(dev_evaluations, model, flatten_result=False))
        return results, data_parameter

    def _make_fold_partition(self, part_names, i):
        data_test_name = part_names[i]
        data_train_names = set(part_names) - set([data_test_name])
        partitions_i = dict()
        if self._separate_dev_test:
            data_dev_name = part_names[(i+1) % len(part_names)]
            data_train_names -= set([data_dev_name])
            for k, p in self._partitions.iteritems():
                partitions_i[k] = p.copy() \
                                    .remove_parts(self._exclude_parts) \
                                    .merge_parts(list(data_train_names), "train") \
                                    .merge_parts([data_dev_name], "dev") \
                                    .merge_parts([data_test_name], "test")
            
        else:
            for k, p in self._partitions.iteritems():
                partitions_i[k] = p.copy() \
                                    .remove_parts(self._exclude_parts) \
                                    .merge_parts(list(data_train_names), "train") \
                                    .merge_parts([data_test_name], "test")
        return partitions_i

class Table:
    def __init__(self, values, row_labels=None, column_labels=None):
        self._values = values
        self._row_labels = row_labels
        self._column_labels = column_labels
    
    def __str__(self):
        s = ''
        if self._column_labels is not None:
            s += '\t'.join(['']+self._column_labels) + '\n'
        if self._row_labels is not None:
            for i, l in enumerate(self._row_labels):
                s += '\t'.join([l]+[str(x) for x in self._values[i]]) + '\n'
        else:
            for i in range(len(self._values)):
                s += '\t'.join([str(x) for x in self._values[i]]) + '\n'

        return s