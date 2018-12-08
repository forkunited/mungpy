import torch
import torch.nn as nn

class BinaryLinearSigmoid(nn.Module):
    def __init__(self):
        super(BinaryLinearSigmoid, self).__init__()
        self._linear = nn.Linear(2, 1)
        self._nl = nn.Sigmoid()

    def forward(self, input1, input2):
        input_mat = torch.cat((input1, input2))
        return self._nl(self._linear(input_mat))

    @classmethod
    def is_reflexive(self):
        return False

    @classmethod
    def is_symmetric(self):
        return True

    @classmethod
    def name(self):
        return "+"

class BinaryProduct(nn.Module):
    def __init__(self):
        super(BinaryProduct, self).__init__()
    
    def forward(self, input1, input2):
        return input1*input2

    @classmethod
    def is_reflexive(self):
        return True

    @classmethod
    def is_symmetric(self):
        return True

    @classmethod
    def name(self):
        return "*"

binary_op_types = [BinaryLinearSigmoid, BinaryProduct]

class GrammarType:
    NONE = "None"
    SELECTING = "Selecting"
    RANDOM = "Random"

class TreeGrammar(nn.Module):
    def __init__(self, opt, input_size, grammar_type, max_binary=800, binary_per_pair=4, extend_interval=100):
        super(TreeGrammar, self).__init__()

        self._input_size = input_size
        self._grammar_type = grammar_type
        self._max_binary = max(input_size, max_binary)
        self._binary_per_pair = binary_per_pair
        self._extend_interval = extend_interval

        self._base_linear = nn.Linear(input_size, 1)
        self._base_linear.weight.data[:,:] = 0.0
        self._base_linear.bias.data[:] = 0.0

        self._binary_ops = dict()
        self._binary_active_count = 0
        self._binary_linears = dict()
        self._binary_indices = dict()
        self._binary_names = []
        for binary_op_type in binary_op_types:
            self._binary_ops[binary_op_type.name()] = nn.ModuleList([binary_op_type() for _ in range(self._max_binary)])
            
            self._binary_linears[binary_op_type.name()] = nn.Linear(self._max_binary, 1)
            self._binary_linears[binary_op_type.name()].weight.data[:,:] = 0.0
            self._binary_linears[binary_op_type.name()].bias.data[:] = 0.0

            self._binary_indices[binary_op_type.name()] = [(-1,-1) for _ in range(self._max_binary)]
            self._binary_names.append(binary_op_type.name())

        self._binary_ops = nn.ModuleDict(self._binary_ops)
        self._binary_linears = nn.ModuleDict(self._binary_linears)

        self._expanded_pairs = set()

    def forward(self, input):
        if self.training and self._opt.get_step() % self._extend_interval == 0:
            self._extend_model()

        batch_size = input.size(0)
        out = self._base_linear(input)
        binary_out = [nn.Variable(torch.zeros(batch_size, self._max_binary)) for i in range(len(self._binary_names))]
        for i in range(self._binary_active_count):
            for j, binary_name in enumerate(self._binary_names):
                index_0, index_1 = self._binary_indices[binary_name][i]
                
                binary_type_0 = index_0 / self._max_binary - 1
                binary_index_0 = index_0 % self._max_binary
                input_0 = None
                if binary_type_0 < 0:
                    input_0 = input[:,binary_index_0]
                else:
                    input_0 = binary_out[binary_type_0][:,binary_index_0]

                binary_type_1 = index_1 / self._max_binary - 1
                binary_index_1 = index_1 % self._max_binary
                input_1 = None
                if binary_type_1 < 0:
                    input_1 = input[:,binary_index_1]
                else:
                    input_1 = binary_out[binary_type_1][:,binary_index_1]

                binary_out[j][:,i] = self._binary_ops[binary_name](input_0, input_1)

        for i in range(len(self._binary_names)):
            out += self._binary_linears[i](binary_out[i])

        return out

    def _extend_model(self):
        for binary_type_0 in range(-1, len(self._binary_names)):
            for binary_type_1 in range(-1, len(self._binary_names)):
                linear_0 = self._base_linear if binary_type_0 < 0 else self._binary_linears[self._binary_names[binary_type_0]]
                linear_1 = self._base_linear if binary_type_1 < 0 else self._binary_linears[self._binary_names[binary_type_1]]
                linear_nonzero_0 = torch.nonzero(torch.abs(linear_0.weight.data.squeeze()) > 0).squeeze()
                linear_nonzero_1 = torch.nonzero(torch.abs(linear_1.weight.data.squeeze()) > 0).squeeze()
                for linear_index_0 in linear_nonzero_0:
                    for linear_index_1 in linear_nonzero_1:
                        index_0 = (binary_type_0 + 1)*self._max_binary + linear_index_0
                        index_1 = (binary_type_1 + 1)*self._max_binary + linear_index_1
                        self._add_ops_for_input_indices(index_0, index_1)

    def _add_ops_for_input_indices(self, index_0, index_1):
        if self._binary_active_count >= self._max_binary:
            return

        for i in range(self._binary_names):
            if index_0 == index_1 and not binary_op_types[i].is_reflexive():
                continue
            if (index_0, index_1) in self._expanded_pairs or \
                (binary_op_types[i].is_symmetric() and (index_1, index_0) in self._expanded_pairs):
                continue

            for j in range(self._binary_active_count, min(self._max_binary, self._binary_active_count + self._binary_per_pair)):
                self._binary_indices[i][self._binary_active_count + j] = (index_0, index_1)

        self._binary_active_count = min(self._max_binary, self._binary_active_count + self._binary_per_pair)
        self._expanded_pairs.add(index_0, index_1)