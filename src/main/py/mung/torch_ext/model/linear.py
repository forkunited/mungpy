import mung.torch_ext.eval
import torch
import torch.nn as nn
from torch.autograd import Variable


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
            raise ValueError("Invalid data parameter: " + str(key))
    
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
        both_out = torch.cat((other_out, out), dim=1)
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
        
class CombinedLoss(nn.Module):
    def __init__(self, losses, weights):
        super(CombinedLoss, self).__init__()
        self._losses = losses
        self._weights = weights

    def forward(inputs, targets):
        total_loss = 0.0
        for i in range(len(inputs)):
            total_loss += self._weights[i]*self._losses[i](inputs[i], targets[i])
        return total_loss

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
        return list((list(self.parameters()))[0].data.view(-1))

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

    def loss(self, batch, data_parameters, loss_criterion):
        output = batch[data_parameters[DataParameter.OUTPUT]]
        if self.on_gpu():
            output = output.cuda()

        model_out = self.forward_batch(batch, data_parameters).squeeze()
        target_out = Variable(output).squeeze()
        if not self._continuous_output:
            target_out = target_out.long()
        return loss_criterion(model_out, target_out)

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
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(LogisticRegression, self).__init__(name, input_size, init_params=init_params, bias=bias)
        self._lrloss = LRLoss(size_average=False)
        self._sigmoid = nn.Sigmoid()

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

    def predict(self, batch, data_parameters, rand=False, include_score=False):
        if rand:
            raise ValueError("Random predictions unsupported by OrdinalLogisticRegression")

        out = self.forward_batch(batch, data_parameters)
        if include_score:
            return torch.sum(out < 0, 1).long(), out[:,0].squeeze().detach()
        else:
            return torch.sum(out < 0, 1).long()

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._olloss)

    def get_loss_criterion(self):
        return self._olloss

class PairwiseLogisticRegression(LinearModel):
    def __init__(self, name, input_size, init_params=None, bias=False):
        super(PairwiseLogisticRegression, self).__init__(name, input_size, init_params=init_params, bias=bias)
        self._lrloss = LRLoss(size_average=False)
        self._sigmoid = nn.Sigmoid()

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
    def __init__(self, name, input_size, label_count, init_params=None, bias=False, alpha=0.5):
        super(PairwiseOrdinalLogisticRegression, self).__init__(name, input_size, output_size=1, init_params=init_params, bias=bias)
        self._theta = nn.Parameter(torch.zeros(label_count-1))
        self._sigmoid = nn.Sigmoid()
        self._alpha = alpha
        self._loss = CombinedLoss(\
            [LRLoss(size_average=False), OrdinalLogisticLoss(label_count), OrdinalLogisticLoss(label_count)],\
            [1.0-alpha, 0.5*self._alpha, 0.5*self._alpha])

    def forward(self, input):
        K = self._theta.size(0)
        if isinstance(input, tuple):
            B = input[0].size(0)
            out_pair = self._linear(input[0] - input[1])
            out_ord0 = self._theta.unsqueeze(0).expand(B, K) - self._linear(input[0]).squeeze().unsqueeze(1).expand(B, K)
            out_ord1 = self._theta.unsqueeze(0).expand(B, K) - self._linear(input[1]).squeeze().unsqueeze(1).expand(B, K)
            return [out_pair, out_ord0, out_ord1]
        else:
            B = input.size(0)            
            return self._theta.unsqueeze(0).expand(B, K) - self._linear(input).squeeze().unsqueeze(1).expand(B, K)

    def loss(self, batch, data_parameters, loss_criterion):
        output = batch[data_parameters[DataParameter.OUTPUT]]
        ordinal = Variable(batch[data_parameters[DataParameter.ORDINAL]])
        if self.on_gpu():
            output = output.cuda()
            ordinal = ordinal.cuda()

        model_out = self.forward_batch(batch, data_parameters).squeeze()
        target_out = Variable(output).squeeze().long()
        ordinal_out = Variable(ordinal).squeeze().long()
        return loss_criterion(model_out, [target_out, ordinal_out[0], ordinal_out[1]])

    def predict(self, batch, data_parameters, rand=False, include_score=False):
        if rand:
            raise ValueError("Random predictions unsupported by PairwiseOrdinalLogisticRegression")

        out = self.forward_batch(batch, data_parameters, include_ordinal=False)
        if isinstance(out, list): # FIXME Currently a hack to check whether predicting over pair or singleton
            p = self._sigmoid(self.forward_batch(batch, data_parameters))
            return torch.bernoulli(p)
        else:
            if include_score:
                return torch.sum(out < 0, 1).long(), out[:,0].squeeze().detach()
            else:
                return torch.sum(out < 0, 1).long()

    def default_loss(self, batch, data_parameters):
        return self.loss(batch, data_parameters, self._loss)

    def get_loss_criterion(self):
        return self._loss