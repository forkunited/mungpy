import abc
import torch
import numpy as np

from mung.data import DataSet
from mung.eval import Evaluation
from torch.autograd import Variable


EVALUATION_BATCH_SIZE = 64 #16 #10 #100

class ModuleEvaluation(Evaluation):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, data, data_parameters, trials=1, batch_indices=False):
        super(ModuleEvaluation, self).__init__(name, data, data_parameters, trials=trials)
        self._batch_indices = batch_indices

    def run(self, model):
        if self._trials == 1:
            return self._run_once(model)
        else:
            results = np.zeros(self._trials)
            for i in range(self._trials):
                results[i] = self._run_once(model)
            return np.mean(results), np.std(results)

    def _run_once(self, model):
        model.eval()

        batch_size = EVALUATION_BATCH_SIZE
        if batch_size > self._data.get_size():
            batch_size = self._data.get_size()

        result = self._initialize_result()

        for i in range(self._data.get_num_batches(batch_size)):
            result = self._aggregate_batch(result, self._run_batch(model, self._data.get_batch(i, batch_size, return_indices=self._batch_indices)))

        final_batch = self._data.get_final_batch(batch_size, return_indices=self._batch_indices)
        if final_batch is not None:
            result = self._aggregate_batch(result, self._run_batch(model, final_batch))

        model.train()

        return self._finalize_result(result)

    @abc.abstractmethod
    def _run_batch(self, model, batch):
        """ Evaluates the model on a given batch of data """

    @abc.abstractmethod
    def _aggregate_batch(self, agg, batch_result):
        """ Aggregates batch result into total """

    @abc.abstractmethod
    def _initialize_result(self):
        """ Initializes the aggregate result """

    @abc.abstractmethod
    def _finalize_result(self, result):
        """ Returns the final value given the aggregate result """

class Loss(ModuleEvaluation):
    def __init__(self, name, data, data_parameters, loss_criterion, norm_dim=False, trials=1):
        super(Loss, self).__init__(name, data, data_parameters, trials=trials)
        self._loss_criterion = loss_criterion
        self._norm_dim = norm_dim

    def _run_batch(self, model, batch):
        loss = model.loss(batch, self._data_parameters, self._loss_criterion)
        if self._norm_dim:
            return (loss[0].data[0], loss[1].data[0])
        else:
            return loss.data[0]

    def _aggregate_batch(self, agg, batch_result):
        if self._norm_dim:
            return (agg[0] + batch_result[0], agg[1] + batch_result[1])
        else:
            return agg + batch_result

    def _initialize_result(self):
        if self._norm_dim:
            return (0.0, 0.0)
        else:
            return 0.0

    def _finalize_result(self, result):
        if self._norm_dim:
            return result[0] / result[1]
        else:
            return result / self._data.get_size()

class DistributionAccuracy(ModuleEvaluation):
    def __init__(self, name, data, data_parameters, target_parameter, model_fn=None, target_indexed = False, check_unique = False, trials=1):
        super(DistributionAccuracy, self).__init__(name, data, data_parameters, trials=trials)
        self._target_parameter = target_parameter
        self._model_fn = model_fn
        self._target_indexed = target_indexed
        self._check_unique = check_unique

    def _run_batch(self, model, batch):
        dist = None
        if self._model_fn is None:
            dist = model.forward_batch(batch, self._data_parameters)
        else:
            dist = self._model_fn(batch, model, self._data_parameters)

        target = batch[self._data_parameters[self._target_parameter]].squeeze()

        model_ps = dist.p().data.cpu()
        max_ps, max_index = torch.max(model_ps, 1, keepdim=True)

        # Indicators of whether maxima are unique
        if self._check_unique:
            max_unique = (torch.sum(max_ps.expand_as(model_ps) == model_ps, 1, keepdim=True) == 1).long()
        else:
            max_unique = torch.ones(target.size(0)).long()

        total_correct = None
        if self._target_indexed:
            total_correct = torch.sum(max_unique.squeeze()*((target == max_index.squeeze()).long()))
        else:
            target_index, has_missing, mask = dist.get_index(target)
            # Count of where model max is same as target
            total_correct = torch.sum(mask*max_unique.squeeze()*((target_index == max_index.squeeze()).long()))

        return float(total_correct)

    def _aggregate_batch(self, agg, batch_result):
        return agg + batch_result

    def _initialize_result(self):
        return 0.0

    def _finalize_result(self, result):
        return result / self._data.get_size()


class ModelStatistic(ModuleEvaluation):
    def __init__(self, name, data, data_parameters, stat_fn, trials=1):
        super(ModelStatistic, self).__init__(name, data, data_parameters, trials=trials)
        self._stat_fn = stat_fn

    def _run_once(self, model):
        model.eval()
        stat = self._stat_fn(model)
        model.train()
        return stat

    def _run_batch(self, model, batch):
        pass

    def _aggregate_batch(self, agg, batch_result):
        pass

    def _initialize_result(self):
        pass

    def _finalize_result(self, result):
        pass

class ModulePrediction(ModuleEvaluation):
    def __init__(self, name, data, data_parameters, target_parameter, rand=False, metrics=[], include_data=False, include_score=False):
        super(ModulePrediction, self).__init__(name, data, data_parameters)
        self._target_parameter = target_parameter
        self._rand = rand
        self._metrics = metrics
        self._include_data = include_data
        self._include_score = include_score

    def get_metrics(self):
        return self._metrics

    def _run_batch(self, model, batch):
        pred = model.predict(batch, self._data_parameters, rand=self._rand, include_score=True)
        return pred[0].numpy(), pred[1].numpy()

    def _aggregate_batch(self, agg, batch_result):
        return np.concatenate((agg[0], batch_result[0]), axis=0), np.concatenate((agg[1], batch_result[1]), axis=0)

    def _initialize_result(self):
        return np.array([]), np.array([])

    def _finalize_result(self, result):
        batch_full = self._data.get_batch(0,self._data.get_size())
        y_true = batch_full[self._data_parameters[self._target_parameter]].squeeze().numpy()

        data_set = self._data.get_data()
        data_set = DataSet(data=list(data_set.get_data()), id_key=data_set.get_id_key())

        y_pred, score = result
        final_result = None
        if len(self._metrics) == 0:
            final_result = [y_true, y_pred]
        else:
            if len(self._metrics) > 1:
                final_result = [metric.compute(y_true, y_pred) for metric in self._metrics]
            else:
                final_result = [self._metrics[0].compute(y_true, y_pred)]

        if self._include_data:
            final_result.append(data_set)
        if self._include_score:
            final_result.append(score)

        if len(final_result) > 1:
            return tuple(final_result)
        else:
            return final_result[0]
