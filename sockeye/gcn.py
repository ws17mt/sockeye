"""
Implementation of Graph Convolutional Network.
Trying to follow the structure of rnn_cell.py in the mxnet code.
"""

import mxnet as mx

def get_gcn(prefix: str):
    gcn = GCNCell()
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
    def __init__(self, prefix='gcn_', params=None, activation='tanh'):
        if params is None:
            params = GCNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._prefix = prefix
        self._params = params
        self._modified = False
        self.reset()
        self._activation = activation
        self._W = self._params.get('W_matrix')
        self._b = self._params.get('bias')

    def convolve(self, adj, inputs):
        outputs = mx.sym.batch_dot(adj, inputs)
        return outputs

    def reset(self):
        pass
