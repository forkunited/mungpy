import time
import copy
import torch.nn.utils
from itertools import ifilter
from torch.optim import Adam, Adadelta
from mung.torch_ext.optim import Adagrad
from mung.eval import Evaluation

class OptimizerType:
    ADAM = "ADAM"
    ADADELTA = "ADADELTA"
    ADAGRAD_MUNG = "ADAGRAD_MUNG"

class Trainer:
    def __init__(self, data_parameters, loss_criterion, logger, evaluation, other_evaluations=None, max_evaluation=False, sample_with_replacement=False,
        batch_size=100, optimizer_type=OptimizerType.ADAM, lr=0.001, grad_clip=None, weight_decay=0.0, log_interval=100, best_part_fn=None, l1_C=0):
        self._data_parameters = data_parameters
        self._loss_criterion = loss_criterion
        self._logger = logger
        self._all_evaluations = [evaluation]
        self._max_evaluation = max_evaluation
        self._sample_with_replacement = sample_with_replacement
        self._batch_size = batch_size
        self._optimizer_type = optimizer_type
        self._lr = lr
        self._grad_clip = grad_clip
        self._weight_decay = weight_decay
        self._log_interval = log_interval
        self._best_part_fn = best_part_fn
        self._l1_C = l1_C

        if other_evaluations is not None:
            self._all_evaluations.extend(other_evaluations)

    def train(self, model, data, iterations):
        model.train()
        start_time = time.time()

        optimizer = None
        if self._optimizer_type == OptimizerType.ADAM:
            optimizer = Adam(ifilter(lambda p: p.requires_grad, model.parameters()), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optimizer_type == OptimizerType.ADAGRAD_MUNG:
            optimizer = Adagrad(ifilter(lambda p: p.requires_grad, model.parameters()), lr=self._lr, lr_decay=0, weight_decay=self._weight_decay, l1_C=self._l1_C, no_non_singleton_l1=False)
        else:
            optimizer = Adadelta(ifilter(lambda p: p.requires_grad, model.parameters()), rho=0.95, lr=self._lr, weight_decay=self._weight_decay)

        set_optimizer = getattr(model, "set_optimizer", None)
        if callable(set_optimizer):
            set_optimizer(optimizer)

        total_loss = 0.0

        self._logger.set_key_order(["Model", "Iteration", "Avg batch time", "Evaluation time", "Avg batch loss"])

        best_part = None
        if self._best_part_fn is None:
            best_part = copy.deepcopy(model)
        else:
            best_part = copy.deepcopy(self._best_part_fn(model))

        best_result = float("inf")
        if self._max_evaluation:
            best_result = float("-inf")
        best_iteration = 0

        # Initial evaluation
        eval_start_time = time.time()
        results = Evaluation.run_all(self._all_evaluations, model)
        results["Model"] = model.get_name()
        results["Iteration"] = 0
        results["Avg batch time"] = 0.0
        results["Evaluation time"] = time.time()-eval_start_time
        results["Avg batch loss"] = 0.0
        self._logger.log(results)
        self._logger.save()

        if not self._sample_with_replacement:
            data.shuffle()

        b = 0
        for i in range(1, iterations + 1):
            batch = None
            if self._sample_with_replacement:
                batch = data.get_random_batch(self._batch_size)
            else:
                batch = data.get_batch(b, self._batch_size)

            loss = model.loss(batch, self._data_parameters, self._loss_criterion)

            optimizer.zero_grad()
            loss.backward()

            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), self._grad_clip)
            
            optimizer.step()

            total_loss += loss.item()

            b += 1

            if (not self._sample_with_replacement) and  b == data.get_num_batches(self._batch_size):
                b = 0
                data.shuffle()

            if i % self._log_interval == 0:
                avg_loss = total_loss / self._log_interval

                avg_batch_ms = (time.time() - start_time)/self._log_interval
                eval_start_time = time.time()
                results = Evaluation.run_all(self._all_evaluations, model)
                results["Model"] = model.get_name()
                results["Iteration"] = i
                results["Avg batch time"] = avg_batch_ms
                results["Evaluation time"] = time.time()-eval_start_time
                results["Avg batch loss"] = avg_loss
                self._logger.log(results)
                self._logger.save()

                main_result = results[self._all_evaluations[0].get_name()]
                if (self._max_evaluation and main_result > best_result) or \
                    ((not self._max_evaluation) and main_result < best_result):
                    best_result = main_result
                    if self._best_part_fn is None:
                        best_part = copy.deepcopy(model)
                    else:
                        best_part = copy.deepcopy(self._best_part_fn(model))
                    best_iteration = i

                total_loss = 0.0
                start_time = time.time()

        model.eval()

        return model, best_part, best_iteration
