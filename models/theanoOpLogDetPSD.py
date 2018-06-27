'''
Theano Op for log-determinant of PSD matrices

@author Wolfgang Roth
'''

import numpy
import theano.tensor as T
from numpy.linalg import slogdet
from theano.gof import Op, Apply

class LogDet_PSD(Op):
    """
    Compute the logarithm of the determinant of a positive semidefinite matrix
    using the Cholesky factorization.
    Note: We implement a separate Node/Op since using the Theano built-in
      Cholesky factorization results in runtime errors (CULA not available)
    """
    def make_node(self, x):
        x = T.as_tensor_variable(x)
        o = T.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            _, ldet = slogdet(x)
            z[0] = numpy.asarray(ldet, dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise
    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * T.nlinalg.matrix_inverse(x).T]
    def __str__(self):
        return "LogDet_PSD"
logdet_psd = LogDet_PSD()
