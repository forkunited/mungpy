import mung.torch_ext.eval
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from mung.torch_ext.model.grammar import TreeGrammar

class DataParameter:
    INPUT = "input"
    OUTPUT = "output"
    ORDINAL = "ordinal"

    def __init__(self, input, output, ordinal="ordinal"):
        self._input = input
        self._output = output
        self._ordinal = ordinal

    def __getitem__(self, key):
        if key == DataParameter.INPUT:
            return self._input
        elif key == DataParameter.OUTPUT:
            return self._output
        elif key == DataParameter.ORDINAL:
            return self._ordinal
        else:
            return key #raise ValueError("Invalid data parameter: " + str(key))
    
    @staticmethod
    def make(input="input", output="output", ordinal="ordinal"):
        return DataParameter(input, output, ordinal=ordinal)

class LRLoss(nn.Module):
    def __init__(self, size_average=True):
        super(LRLoss, self).__init__()
        self._nllloss = nn.NLLLoss(size_average=size_average)
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, input, target):
        out = self._log_sigmoid(input)
        other_out = self._log_sigmoid(-input)
        both_out = torch.cat((other_out.unsqueeze(1), out.unsqueeze(1)), dim=1)
        return self._nllloss(both_out, target.squeeze().long())

# See http://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
class OrdinalLogisticLoss(nn.Module):
    def __init__(self, label_count):
        super(OrdinalLogisticLoss, self).__init__()
        self._ordinals = torch.arange(0, label_count - 1)
        self._log_sigmoid = nn.LogSigmoid()

    def forward(self, input, target):
        # input is (Batch size) x (K-1) (labels)
        # target is (Batch size)
        B = target.size(0)
        K = self._ordinals.size(0)

        l = self._ordinals.unsqueeze(0).expand(B, K)
        y = target.unsqueeze(1).expand(B, K)
        
        # Compute s(l;y)
        s = l - y 
        s = (s >= 0).float()-(s < 0).float()

        # Compute all-threshold loss
        return -torch.sum((y >= 0).float()*self._log_sigmoid(s * input))

        # FIXME Remove below here (scratch attempt at hinge)
        #l_in = (s * input).unsqueeze(2)
        #zero = Variable(torch.zeros(l_in.size()))
        #l_in = torch.cat((l_in, zero), 2)
        #return torch.sum((y >= 0).float()*torch.max(l_in, 2)[0])

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super(CombinedLoss, self).__init__()
        self._losses = losses
        self._weights = weights

    def forward(self, inputs, targets):
        total_loss = 0.0
        for i in range(len(inputs)):
            total_loss += self._weights[i]*self._losses[i](inputs[i], targets[i])
        return total_loss

class DisjunctiveLoss(nn.Module):
    def __init__(self, losses, loss_rescalings=None):
        super(DisjunctiveLoss, self).__init__()
        self._losses = losses
        if loss_rescalings is None:
            self._loss_rescalings = [1.0]*len(self._losses)
        else:
            self._loss_rescalings = loss_rescalings

    def forward(self, input, target, loss_index):
        return self._loss_rescalings[loss_index]*self._losses[loss_index](input, target)

class LinearModel(nn.Module):
    def __init__(self, name, input_size, output_size=1, init_params=None, bias=False, continuous_output=False):
        super(LinearModel, self).__init__()
        self._name = name

        self._input_size = input_size
        self._output_size = output_size
        self._linear = nn.Linear(self._input_size, self._output_size, bias=bias)
        self._continuous_output = continuous_output
        if init_params is not None:
            self._linear.weight = nn.Parameter(init_params.unsqueeze(0))
        else:
            self._linear.weight = nn.Parameter(torch.zeros(self._output_size,self._input_size))

    def get_name(self):
        return self._name

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def get_weights(self):
        params = list(self.parameters())
        for param in params:
            weights = param.data.view(-1)
            if weights.size(0) ==  self._input_size * self._output_size: # FIXME Hack for now
                return [w.item() for w in list(weights)] 
        return None

    def get_nonzero_weight_count(self):
        nz = torch.nonzero((list(self.parameters()))[0].data.view(-1))
        if len(nz.size()) == 0:
            return 0
        else:
            return nz.size(0)

    def get_bias(self):
        return list(self.parameters())[1].data[0]

    def forward(self, input):
        return self._linear(input)

    def forward_batch(self, batch, data_parameters):
        input = Variable(batch[data_parameters[DataParameter.INPUT]])
        if self.on_gpu():
            input = input.cuda()
        return self(input)

    def score(self, batch, data_parameters):
        out = self.forward_batch(batch, data_parameters)
        return out.squeeze(1).detach()

    def loss(self, batch, data_parameters, loss_criterion):
        output = batch[data_parameters[DataParameter.OUTPUT]]
        if self.on_gpu():
            output = output.cuda()

        model_out = self.forward_batch(batch, data_parameters).squeeze()
        target_out = Variable(output).squeeze()
        if not self._continuous_output:
            target_out = target_out.long()
        return loss_criterion(model_out, target_out)

    def predict_data(self, data, data_parameters, rand=False, prediction_batch_size=64):
        pred = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            cur_pred = self.predict(batch, data_parameters, rand=rand).detach().numpy()
            if pred is None:
                pred = cur_pred
            else:
                pred = np.concatenate((pred, cur_pred), axis=0)
        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            cur_pred = self.predict(batch, data_parameters, rand=rand).detach().numpy()
            if pred is None:
                pred = cur_pred
            else:
                pred = np.concatenate((pred, cur_pred), axis=0)
        return pred

    def score_data(self, data, data_parameters, prediction_batch_size=64):
        score = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            cur_score = self.score(batch, data_parameters).numpy()
            if score is None:
                score = cur_score
            else:
                score = np.concatenate((score, cur_score), axis=0)

        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            cur_score = self.score(batch, data_parameters).numpy()
            if score is None:
                score = cur_score
            else:
                score = np.concatenate((score, cur_score), axis=0)
        return score

    def p_data(self, data, data_parameters, prediction_batch_size=64):
        p = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            if p is None:
                p = self.p(batch, data_parameters).numpy()
            else:
                p = np.concatenate((p, self.p(batch, data_parameters).numpy()), axis=0)
        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            p = np.concatenate((p, self.p(batch, data_parameters).numpy()), axis=0)
        return p

    def predict(self, batch, data_parameters, rand=False):
        raise NotImplementedError()

    def p(self, batch, data_parameters):
        raise NotImplementedError()

class LinearRegression(LinearModel):
    def __init__(self, name, input_size, init_params=None, bias=False, std=1.0):
        super(LinearRegression, self).__init__(name, input_size, init_params=init_params, 
            bias=bias, continuous_output=True)
        self._mseloss = nn.MSELoss(size_average=False)
        self._std = std

    def predict(self, batch, data_parameters, rand=False):
        if not rand:
            return self.forward_batch(batch, data_parameters).detach().squeeze()
        else:
            mu = self.forward_batch(batch, data_parameters)
            return torch.normal(mu, self._std)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._mseloss)

    def get_loss_criterion(self):
        return self._mseloss

class LogisticRegression(LinearModel):
    def __init__(self, name, input_size, output_size=1, init_params=None, bias=False, hidden_sizes=[]):
        super(LogisticRegression, self).__init__(name, input_size if len(hidden_sizes) == 0 else hidden_sizes[-1], output_size=output_size, init_params=init_params, bias=bias)
        self._lrloss = LRLoss(size_average=False)
        self._sigmoid = nn.Sigmoid()

        self._hidden_nl = nn.LeakyReLU()
        self._hidden_layers = []
        cur_size = input_size
        for hidden_size in hidden_sizes:
            self._hidden_layers.append(nn.Linear(cur_size, hidden_size, bias=bias))
            cur_size = hidden_size

    def forward(self, input):
        for hidden_layer in self._hidden_layers:
            input = self._hidden_nl(hidden_layer(input))
        return self._linear(input)

    def predict(self, batch, data_parameters, rand=False):
        p = self._sigmoid(self.forward_batch(batch, data_parameters))
        if not rand:
            return p > 0.5
        else:
            return torch.bernoulli(p)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._lrloss)

    def get_loss_criterion(self):
        return self._lrloss

# See http://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
class OrdinalLogisticRegression(LinearModel):
    def __init__(self, name, input_size, label_count, init_params=None, bias=False):
        super(OrdinalLogisticRegression, self).__init__(name, input_size, output_size=1, init_params=init_params, bias=bias)
        self._theta = nn.Parameter(torch.zeros(label_count-1))
        self._olloss = OrdinalLogisticLoss(label_count)

    def forward(self, input):
        B = input.size(0)
        K = self._theta.size(0)
        return self._theta.unsqueeze(0).expand(B, K) - self._linear(input).squeeze().unsqueeze(1).expand(B, K)

    def predict(self, batch, data_parameters, rand=False):
        if rand:
            raise ValueError("Random predictions unsupported by OrdinalLogisticRegression")

        out = self.forward_batch(batch, data_parameters)
        return torch.sum(out < 0, 1).long()

    def score(self, batch, data_parameters):
        out = self.forward_batch(batch, data_parameters)
        return out[:,0].squeeze().detach()

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._olloss)

    def get_loss_criterion(self):
        return self._olloss

class PairwiseLogisticRegression(LinearModel):
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(PairwiseLogisticRegression, self).__init__(name, input_size, init_params=init_params, bias=bias)
        self._lrloss = LRLoss(size_average=False)
        self._sigmoid = nn.Sigmoid()

    def forward_batch(self, batch, data_parameters):
        input = batch[data_parameters[DataParameter.INPUT]]
        if self.on_gpu():
            input = input[0].cuda(), input[1].cuda()
        input = Variable(input[0]), Variable(input[1])
        return self(input)

    def forward(self, input):
        return self._linear(input[0] - input[1])

    def predict(self, batch, data_parameters, rand=False):
        p = self._sigmoid(self.forward_batch(batch, data_parameters))
        if not rand:
            return p > 0.5
        else:
            return torch.bernoulli(p)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._lrloss)

    def get_loss_criterion(self):
        return self._lrloss

# See http://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
class OrdisticRegression(LinearModel):
    def __init__(self, name, input_size, label_count, init_params=None, bias=False):
        super(OrdisticRegression, self).__init__(name, input_size, output_size=1, init_params=init_params, bias=bias)
        self._mu = nn.Parameter(torch.zeros(label_count))
        self._celoss = nn.CrossEntropyLoss(reduction='sum')
        self._softmax = nn.Softmax()
        self._one = torch.ones(1)

    def forward(self, input):
        z = self._linear(input)
        
        B = z.size(0)
        K = self._mu.size(0)

        #mu = torch.cat((-self._one *0.0, self._mu_inner, self._one * 4.0)) 
        mu = self._mu.unsqueeze(0).expand(B, K)
        z = z.squeeze().unsqueeze(1).expand(B, K) 
        
        return mu * z - (mu ** 2.0) / 2.0

    def predict(self, batch, data_parameters, rand=False):
        p = self._softmax(self.forward_batch(batch, data_parameters))
        if not rand:
            return torch.argmax(p, dim=1)
        else:
            return torch.multinomial(p)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._celoss)

    def get_loss_criterion(self):
        return self._celoss

class MultinomialLogisticRegression(LinearModel):
    def __init__(self, name, input_size, label_count, init_params=None, bias=False):
        super(MultinomialLogisticRegression, self).__init__(name, input_size, output_size=label_count, init_params=init_params, bias=bias)

        self._celoss = nn.CrossEntropyLoss(reduction='sum')
        self._softmax = nn.Softmax(dim=1)

    def predict(self, batch, data_parameters, rand=False):
        p = self._softmax(self.forward_batch(batch, data_parameters))
        if not rand:
            return torch.argmax(p, dim=1)
        else:
            return torch.multinomial(p)

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._celoss)

    def get_loss_criterion(self):
        return self._celoss


class PairwiseOrdinalLogisticRegression(LinearModel):
    def __init__(self, name, input_size, label_count, init_params=None, bias=False, ordinal_rescaling=1.0, confidence_ordinals=False, constant_scalar=True, hidden_sizes=[]):
        super(PairwiseOrdinalLogisticRegression, self).__init__(name, input_size if len(hidden_sizes) == 0 else hidden_sizes[-1], output_size=1, init_params=init_params, bias=bias)
        self._theta = nn.Parameter(torch.zeros(label_count-1))
        self._sigmoid = nn.Sigmoid()
        self._loss = DisjunctiveLoss([LRLoss(size_average=False), OrdinalLogisticLoss(label_count)], \
                                     [1.0, ordinal_rescaling])
        self._confidence_ordinals = confidence_ordinals
        self._constant_scalar = constant_scalar

        self._hidden_nl = nn.LeakyReLU()
        self._hidden_layers = []
        cur_size = input_size
        for hidden_size in hidden_sizes:
            self._hidden_layers.append(nn.Linear(cur_size, hidden_size, bias=bias))
            cur_size = hidden_size

        if not constant_scalar:
            self._s = nn.Parameter(torch.ones(label_count-1))

    def _is_ordinal_batch(self, batch, data_parameters):
        return not isinstance(batch[data_parameters[DataParameter.INPUT]], tuple)

    def get_theta(self):
        return [t.item() for t in list(self._theta)] 

    def forward_batch(self, batch, data_parameters):
        input = batch[data_parameters[DataParameter.INPUT]]
        if self._is_ordinal_batch(batch, data_parameters):
            if self.on_gpu():
                input = input.cuda()
            input = Variable(input)
        else:
            if self.on_gpu():
                input = input[0].cuda(), input[1].cuda()
            input = Variable(input[0]), Variable(input[1])
        return self(input)

    def forward(self, input):
        if not isinstance(input, tuple):
            B = input.size(0)      
            K = self._theta.size(0)
            if self._constant_scalar:
                return self._theta.unsqueeze(0).expand(B, K) - self._forward_through_layers(input).expand(B, K)
            else:
                return (self._theta.unsqueeze(0).expand(B, K) - self._forward_through_layers(input).expand(B, K))/self._s.unsqueeze(0).expand(B, K)
        else:
            return self._forward_through_layers(input)

    def _forward_through_layers(self, input):
        if isinstance(input, tuple):
            input = input[0] - input[1]
        for hidden_layer in self._hidden_layers:
            input = self._hidden_nl(hidden_layer(input))
        return self._linear(input)

    def loss(self, batch, data_parameters, loss_criterion):
        output = None
        
        if self._is_ordinal_batch(batch, data_parameters):
            output = batch[data_parameters[DataParameter.ORDINAL]]
        else:
            output = batch[data_parameters[DataParameter.OUTPUT]]
        if self.on_gpu():
            output = output.cuda()

        model_out = self.forward_batch(batch, data_parameters).squeeze()
        target_out = Variable(output).squeeze().long()

        if self._is_ordinal_batch(batch, data_parameters):
            return loss_criterion(model_out, target_out, 1)
        else:
            return loss_criterion(model_out, target_out, 0)

    def predict(self, batch, data_parameters, rand=False):
        if rand:
            raise ValueError("Random predictions unsupported by PairwiseOrdinalLogisticRegression")

        out = self.forward_batch(batch, data_parameters)
        if self._is_ordinal_batch(batch, data_parameters):
            if self._confidence_ordinals:
                p = self._sigmoid(out)
                K = p.size(1)+1
                p = torch.cat((p, (1.0-p[:,K-2]).unsqueeze(1)), 1)
                p[:,1:(K-1)] = p[:,1:(K-1)] - p[:,0:(K-2)]
                return torch.argmax(p, dim=1)
            else:
                return torch.sum(out < 0, 1).long()
        else:
            p = self._sigmoid(out)
            return torch.bernoulli(p).squeeze(1)

    def p(self, batch, data_parameters):
        out = self.forward_batch(batch, data_parameters)
        pr = self._sigmoid(out)
        K = pr.size(1)+1
        pr = torch.cat((pr, (1.0-pr[:,K-2]).unsqueeze(1)), 1)
        pr[:,1:(K-1)] = pr[:,1:(K-1)] - pr[:,0:(K-2)]
        return pr.detach()

    def score(self, batch, data_parameters):
        out = self.forward_batch(batch, data_parameters)
        if self._is_ordinal_batch(batch, data_parameters):
            return -out[:,0].detach() #.squeeze(0).detach()
        else:
            return out.squeeze(1).detach()

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._loss)

    def get_loss_criterion(self):
        return self._loss


# See http://ttic.uchicago.edu/~nati/Publications/RennieSrebroIJCAI05.pdf
class OrdinalTreeGrammarRegression(nn.Module):
    def __init__(self, name, input_size, label_count, grammar_type, opt, 
                 max_binary=800, binary_per_pair=4, extend_interval=100):
        super(OrdinalTreeGrammarRegression, self).__init__()
        self._theta = nn.Parameter(torch.zeros(label_count-1))
        self._olloss = OrdinalLogisticLoss(label_count)

        self._name = name
        self._input_size = input_size
        self._output_size = 1
        self._grammar = TreeGrammar(opt, self._input_size, grammar_type, max_binary=max_binary, binary_per_pair=binary_per_pair, extend_interval=extend_interval)

    def set_optimizer(self, opt):
        self._grammar.set_optimizer(opt)

    def forward(self, input):
        B = input.size(0)
        K = self._theta.size(0)
        return self._theta.unsqueeze(0).expand(B, K) - self._grammar(input).squeeze().unsqueeze(1).expand(B, K)

    def predict(self, batch, data_parameters, rand=False):
        if rand:
            raise ValueError("Random predictions unsupported by OrdinalLogisticRegression")

        out = self.forward_batch(batch, data_parameters)
        return torch.sum(out < 0, 1).long()

    def score(self, batch, data_parameters):
        out = self.forward_batch(batch, data_parameters)
        return out[:,0].squeeze().detach()

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._olloss)

    def get_loss_criterion(self):
        return self._olloss

    def get_name(self):
        return self._name

    def on_gpu(self):
        return next(self.parameters()).is_cuda

    def get_weights(self):
        # FIXME
        #params = list(self.parameters())
        #for param in params:
        #    weights = param.data.view(-1)
        #    if weights.size(0) ==  self._input_size * self._output_size: # FIXME Hack for now
        #        return [w.item() for w in list(weights)] 
        return None

    def get_nonzero_weight_count(self):
        return 0
        # FIXME
        #nz = torch.nonzero((list(self.parameters()))[0].data.view(-1))
        #if len(nz.size()) == 0:
        #    return 0
        #else:
        #    return nz.size(0)

    def get_bias(self):
        return None
        # FIXME return list(self.parameters())[1].data[0]

    def forward_batch(self, batch, data_parameters):
        input = Variable(batch[data_parameters[DataParameter.INPUT]])
        if self.on_gpu():
            input = input.cuda()
        return self(input)

    def loss(self, batch, data_parameters, loss_criterion):
        output = batch[data_parameters[DataParameter.ORDINAL]]
        if self.on_gpu():
            output = output.cuda()

        model_out = self.forward_batch(batch, data_parameters).squeeze()
        target_out = Variable(output).squeeze().long()
        return loss_criterion(model_out, target_out)

    def predict_data(self, data, data_parameters, rand=False, prediction_batch_size=64):
        pred = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            cur_pred = self.predict(batch, data_parameters, rand=rand).numpy()
            if pred is None:
                pred = cur_pred
            else:
                pred = np.concatenate((pred, cur_pred), axis=0)
        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            cur_pred = self.predict(batch, data_parameters, rand=rand).numpy()
            if pred is None:
                pred = cur_pred
            else:
                pred = np.concatenate((pred, cur_pred), axis=0)
        return pred

    def score_data(self, data, data_parameters, prediction_batch_size=64):
        score = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            cur_score = self.score(batch, data_parameters).numpy()
            if score is None:
                score = cur_score
            else:
                score = np.concatenate((score, cur_score), axis=0)

        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            cur_score = self.score(batch, data_parameters).numpy()
            if score is None:
                score = cur_score
            else:
                score = np.concatenate((score, cur_score), axis=0)
        return score

    def p_data(self, data, data_parameters, prediction_batch_size=64):
        p = None
        prediction_batch_size = min(prediction_batch_size, data.get_size())
        for i in range(data.get_num_batches(prediction_batch_size)):
            batch = data.get_batch(i, prediction_batch_size)
            if p is None:
                p = self.p(batch, data_parameters).numpy()
            else:
                p = np.concatenate((p, self.p(batch, data_parameters).numpy()), axis=0)
        batch = data.get_final_batch(prediction_batch_size)
        if batch is not None:
            p = np.concatenate((p, self.p(batch, data_parameters).numpy()), axis=0)
        return p
