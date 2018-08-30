import abc
import numpy as np
import sklearn.metrics
import scipy.stats
from mung.data import DataSet

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
        return ClassificationReportResult(sklearn.metrics.classification_report(y_true.astype(int), y_pred.astype(int), 
            self._labels, self._target_names, self._sample_weight, self._digits))

class ClassificationReportResult:
    def __init__(self, result_str):
        self._label_result = dict()
        self._aggregate_result = dict()
        self._parse_result_str(result_str)
        self._result_str = result_str

    def __str__(self):
        return self._result_str

    def _parse_result_str(self, result_str):
        lines = result_str.split("\n")
        for i, line in enumerate(lines):
            line = line.strip().replace("avg / total", "avg/total")
            if len(line) == 0 or i == 0:
                continue
            line_parts = [line_part.strip() for line_part in line.split()]
            label, precision, recall, f1, support = line_parts
            if label == "avg/total":
                self._aggregate_result = { "precision" : float(precision), "recall" : float(recall), "f1" : float(f1), "support" : int(support) }
            else:
                self._label_result[label] = { "precision" : float(precision), "recall" : float(recall), "f1" : float(f1), "support" : int(support) }

    def get_single_column_table(self):
        row_labels = []
        values = []

        for label, label_results in self._label_result.iteritems():
            for label_result_name, label_result_value in label_results.iteritems():
                row_labels.append(label_result_name + "_" + label)
                values.append([label_result_value])

        for agg_result_name, agg_result_value in self._aggregate_result.iteritems():
            row_labels.append(agg_result_name)
            values.append([agg_result_value])

        return Table(values, row_labels=row_labels)

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
            dev_evaluation_config, trainer_config, data_metrics, logger):
        part_names = list(set(self._partitions.itervalues().next().get_part_names()) - self._exclude_parts)
        k = len(part_names) # k-fold cv with parts
        results = []
        models = []
        data_parameters = []
        datas = dict()
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

                if key not in datas:
                    datas[key] = []
                datas[key].append(d_parts["test"])

            # Load model and evaluations
            data_parameter, model = self._load_model_fn(model_config, D_params)
            train_evaluations = self._load_evaluation_fn(train_evaluation_config, datas_i, data_parameter)
            dev_evaluations = self._load_evaluation_fn(dev_evaluation_config, datas_i, data_parameter)

            # Train and evaluate on current fold (i)
            _, best_model, _ = self._train_fn(trainer_config, data_parameter, \
                model.get_loss_criterion(), logger, train_evaluations, model, datas_i)

            models.append(best_model)
            data_parameters.append(data_parameter)
            results.append(Evaluation.run_all(dev_evaluations, best_model, flatten_result=False))

        return FoldedCVResults(models, data_parameters, results, datas, data_metrics)

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

class FoldedCVResults:
    def __init__(self, fold_models, data_parameters, fold_evaluations, datas, data_metrics):
        self._fold_models = fold_models
        self._data_parameters = data_parameters
        self._fold_evaluations = fold_evaluations
        self._datas = datas
        self._data_metrics = data_metrics

    def _merge_data(self, data_name):
        full_data = DataSet(data=[])
        for data in self._datas[data_name]:
            full_data = full_data.union(data.get_data())
        return full_data

    def _predict_data(self, data_name):
        y_pred = np.array([])
        data = self._datas[data_name]
        for i in range(self.get_fold_count()):
            y_pred_i = self._fold_models[i].predict_data(data[i], self._data_parameters[i], rand=False)
            y_pred = np.concatenate((y_pred,y_pred_i))
        return y_pred

    def _view_data(self, data_name, target_parameter):
        target = np.array([])
        data = self._datas[data_name]
        for i in range(self.get_fold_count()):
            target_i = data[i].get_batch(0,data[i].get_size())[self._data_parameters[i][target_parameter]].squeeze().numpy()
            target = np.concatenate((target, target_i))
        return target

    def _score_data(self, data_name):
        y_score = np.array([])
        data = self._datas[data_name]
        for i in range(self.get_fold_count()):
            y_score_i = self._fold_models[i].score_data(data[i], self._data_parameters[i])
            y_score = np.concatenate((y_score,y_score_i))
        return y_score

    def get_data_parameter(self, fold_index=0):
        return self._data_parameters[fold_index]

    def get_data_names(self):
        return self._datas.keys()

    def get_fold_count(self):
        return len(self._fold_models)

    def get_prediction_table(self, data_name, datum_str_fn, datum_true_fn, pred_str_fn, datum_type_name="datum"):
        merged_data = self._merge_data(data_name)
        y_pred = self._predict_data(data_name)
        y_score = self._score_data(data_name)
        table_rows = []
        for i in range(merged_data.get_size()):
            table_rows.append([datum_str_fn(merged_data[i]), \
                               datum_true_fn(merged_data[i]), \
                               pred_str_fn(y_pred[i]), \
                               y_score[i]])
        return Table(table_rows, column_labels=[datum_type_name, "y_true", "y_pred", "score"])
            
    def get_fold_evaluation_results(self, i):
        return self._fold_evaluations[i]

    def get_data_metric_results(self, data_name):
        if data_name not in self._data_metrics:
            return None

        prediction_metrics = self._data_metrics[data_name]["prediction_metrics"]
        score_metrics = self._data_metrics[data_name]["score_metrics"]
        target_parameter = self._data_metrics[data_name]["target_parameter"]
        y_pred = self._predict_data(data_name)
        y_score = self._score_data(data_name)
        y_true = self._view_data(data_name, target_parameter)
        results = dict()
        for metric in prediction_metrics:
            results[metric.get_name()] = metric.compute(y_true, y_pred)
        for metric in score_metrics:
            results[metric.get_name()] = metric.compute(y_true, y_score)
        return results

    def get_data_metric_results_string(self, data_name):
        results = self.get_data_metric_results(data_name)
        results_str = ""
        for metric_name, metric_value in results.iteritems():
            if isinstance(metric_value, str) or \
                isinstance(metric_value, ClassificationReportResult) or \
                isinstance(metric_value, Table):
                results_str += "-------------" + metric_name + "-------------\n"
                results_str += str(metric_value)
                results_str += "---------------------------------------------\n"
            else:
                results_str += metric_name + "\t" + str(metric_value) + "\n"
        return results_str

    def get_numerical_metric_results(self, heading=""):
        column_labels = ["Data", "Metric", heading]
        data_labels = []
        metric_labels = []
        metric_values = []
        for data_name in self.get_data_names():
            data_results = self.get_data_metric_results(data_name)
            if data_results is None:
                continue
            for metric_name, metric_value in data_results.iteritems():
                if isinstance(metric_value, Table):
                    continue
                elif isinstance(metric_value, ClassificationReportResult):
                    table = metric_value.get_single_column_table()
                    data_labels.extend([data_name]*len(table.get_row_labels()))
                    metric_labels.extend([metric_name + "_" + row_label for row_label in table.get_row_labels()])
                    metric_values.extend([x for y in table.get_values() for x in y]) # Flatten values
                elif isinstance(metric_value, tuple):
                    data_labels.append(data_name)
                    metric_labels.append(metric_name)
                    metric_values.append(metric_value[0])
                else:
                    data_labels.append(data_name)
                    metric_labels.append(metric_name)
                    metric_values.append(metric_value)
        values = [[data_labels[i], metric_labels[i], metric_values[i]] for i in range(len(metric_labels))]
        return Table(values, column_labels=column_labels)

class Table:
    def __init__(self, values, row_labels=None, column_labels=None):
        self._values = values
        self._row_labels = row_labels
        self._column_labels = column_labels

    def get_row_labels(self):
        return self._row_labels

    def get_column_labels(self):
        return self._column_labels

    def get_values(self):
        return self._values
    
    def __str__(self):
        values = []
        for i in range(len(self._values)):
            values.append([])
            for j in range(len(self._values[i])):
                if isinstance(self._values[i][j],float) or isinstance(self._values[i][j], int):
                    values[i].append(str(self._values[i][j]))
                else:
                    values[i].append(self._values[i][j].encode("utf-8"))

        s = ''
        if self._column_labels is not None:
            column_labels = self._column_labels
            if self._row_labels is not None:
                column_labels = [''] + column_labels
            s += '\t'.join(column_labels) + '\n'
        if self._row_labels is not None:
            for i, l in enumerate(self._row_labels):
                s += '\t'.join([l]+[x for x in values[i]]) + '\n'
        else:
            for i in range(len(self._values)):
                s += '\t'.join([x for x in values[i]]) + '\n'

        return s