import unittest
import torch
import sys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from mung.torch_ext.eval import DistributionAccuracy, DataParameter
from mung.feature import MultiviewDataSet

data_dir = sys.argv[1]
target_dir = sys.argv[2]

# Necessary to allow unittest.main() to work
del sys.argv[1]
del sys.argv[2]

class TestEvaluation(unittest.TestCase):

    def test_accuracy(self):
        class Categorical:
            def __init__(self, vs, ps=None, on_gpu=False, unnorm=False):
                self._vs = vs
                self._ps = ps
                if self._ps is None:
                    vs_for_size = self._vs
                    if isinstance(self._vs, tuple):
                        vs_for_size = self._vs[0]
                    self._ps = torch.ones(vs_for_size.size(0), vs_for_size.size(1))
                    if unnorm:
                        self._ps = Variable(self._ps)
                    else:
                        self._ps = Variable(self._ps/torch.sum(self._ps, dim=1).repeat(1,self._ps.size(1)))
                if on_gpu:
                    self._ps = self._ps.cuda()

            def __getitem__(self, key):
                if key == 0:
                    return self._ps

            def support(self):
                return self._vs

            def p(self):
                return self._ps

            def get_index(self, value):
                index = torch.zeros(value.size(0)).long()
                mask = torch.ones(value.size(0)).long()
                support = self._vs
                
                for b in range(support.size(0)): # Over batch
                    found = False
                    for s in range(support.size(1)): # Over samples in support
                        if (len(value.size()) > 1 and torch.equal(support[b,s], value[b])) \
                        or (len(value.size()) == 1 and support[b,s] == value[b]):
                            index[b] = s
                            found = True
                            break
                    if not found:
                        has_missing = True
                        mask[b] = 0
                return index, has_missing, mask

        class TestModel:
            def __init__(self, acc):
                self._acc = acc

            def eval(self):
                pass

            def train(self):
                pass

            def forward_batch(self, batch, data_parameter):
                targets = batch[data_parameters[DataParameter.TARGET]]
                max_target = int(torch.max(targets))
                err = 1.0 -self._acc
                change_count = err*targets.size(1)
                change_indices = np.random.choice(targets.size(0), change_count, replace=False)
                change_indices = torch.from_numpy(change_indices)
                vs = torch.arange(0, max_target).unsqueeze(0).repeat(targets.size(0),1)
                ps = torch.ones(targets.size(0), max_target)

                for i in range(targets.size(0)):
                    ps[i, targets[i]] *= 2.0

                for i in range(change_indices.size(0)):
                    index = change_indices[i]
                    ps[index, targets[i]] /= 4.0

                return Categorical(vs=vs, ps=ps)


        data = MultiviewDataSet.load(data_dir, dfmat_paths={ "target" : target_dir })
        data_parameters = DataParameter.make(target="target")
        acc = DistributionAccuracy("Accuracy", data, data_parameters)

        for i in range(4):
            model = TestModel(acc=0.25*i)
            result = acc.run(model)
            self.assertEqual(result, 0.25*i)



if __name__ == '__main__':
    unittest.main()