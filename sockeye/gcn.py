"""
Implementation of Graph Convolutional Network.
Trying to follow the structure of rnn_cell.py in the mxnet code.
"""

import mxnet as mx
import sockeye.constants as C

def get_gcn(num_hidden, prefix: str):
    gcn = GCNCell(num_hidden, prefix=prefix)
    return gcn
   

class GCNParams(object):
    """Container to hold GCN variables.
    Used for parameter sharing.

    Parameters
    ----------
    prefix : str
        All variables' name created by this container will
        be prepended with prefix.
    """

    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get a variable with name or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        **kwargs :
            more arguments that's passed to symbol.Variable
        """
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = mx.sym.Variable(name, **kwargs)
        return self._params[name]
    

class GCNCell(object):
    """GCN cell
    """
    def __init__(self, num_hidden, prefix='gcn_', params=None, 
                 activation='tanh'):
        if params is None:
            params = GCNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._num_hidden = num_hidden
        self._prefix = prefix
        self._params = params
        self._modified = False
        self.reset()
        self._activation = activation
        self._W = self._params.get('weight')
        self._b = self._params.get('bias')

    def convolve(self, adj, inputs, seq_len):
        # inputs go through linear transformation
        # annoyingly, MXNet does not have a batched version
        # of FullyConnected so we need some reshaping
        reshaped = mx.symbol.reshape(inputs, (-3, -1))
        outputs = mx.symbol.FullyConnected(data=reshaped, weight=self._W,
                                           bias=self._b, num_hidden=self._num_hidden,
                                           name='%sFC'%self._prefix)
        outputs = mx.symbol.reshape(outputs, (-1, seq_len, self._num_hidden))
        # now they are convolved according to the adj matrix                                  
        outputs = mx.symbol.batch_dot(adj, outputs)
        # finally, we apply a non-linear transformation
        outputs = mx.symbol.Activation(outputs, act_type=self._activation)
        return outputs

    def reset(self):
        pass
