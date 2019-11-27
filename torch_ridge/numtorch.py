# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD 3 clause

import torch
import numpy as np


class NumTorch():
    """Wrapping class to interchangeably use Numpy and Pytorch

    WIP for current needs. Expand using
    https://github.com/torch/torch7/wiki/Torch-for-Numpy-users

    Parameters
    ----------
        X : array | tensor
        used to identify the data type and the device.

    Returns
    -------
        self : a class with numpy-like methods
    """
    def __init__(self, module='numpy', device=None):

        if module == 'torch':

            def array(*args, **kwargs):
                return torch.tensor(*args, device=device, **kwargs)

            def zeros(*args, **kwargs):
                return torch.zeros(*args, device=device, **kwargs)

            def ones(*args, **kwargs):
                return torch.ones(*args, device=device, **kwargs)

            def mean(*args, **kwargs):
                if 'dim' in kwargs.keys():
                    kwargs['axis'] = kwargs['dim']
                    kwargs.pop('dim')
                return torch.mean(*args, **kwargs)

            diag = torch.diag

            def svd(input, full_matrices=True, **kwargs):
                kwargs['some'] = not full_matrices
                U, s, V = torch.svd(input, **kwargs)
                return U, s, V.t()

            def shape(X, axis=None):
                s = X.size()
                if axis is not None:
                    s = X[axis]
                return s

        else:
            array = np.array
            zeros = np.zeros
            diag = np.diag
            shape = np.shape
            mean = np.mean
            ones = np.ones
            svd = np.linalg.svd

        self.array = array
        self.zeros = zeros

        class Linealg(object):
            def __init__(self, svd):
                self.svd = svd

        self.linealg = Linealg(svd)
        self.diag = diag
        self.shape = shape
        self.ones = ones
        self.mean = mean


def assert_tensor_almost_equal(x, y):
    """testing function"""
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    np.testing.assert_array_almost_equal(x, y)


if __name__ == '__main__':
    # Unit tests
    np.random.seed(0)
    Xnp = np.random.randn(100, 10)
    Xth = torch.from_numpy(Xnp)

    np_ = NumTorch('numpy')
    th_ = NumTorch('torch')

    methods = ('array', 'zeros', 'svd', 'diag', 'shape', 'ones', 'mean')

    assert_tensor_almost_equal(np_.zeros((10, 2)), th_.zeros((10, 2)))
    assert_tensor_almost_equal(np_.ones((10, 2)), th_.ones((10, 2)))
    assert_tensor_almost_equal(np_.array(Xnp), th_.array(Xth))
    for var_np, var_th in zip(np_.linealg.svd(Xnp), th_.linealg.svd(Xth)):
        assert_tensor_almost_equal(var_np, var_th)
    assert_tensor_almost_equal(np_.diag(Xnp), th_.diag(Xth))
    assert_tensor_almost_equal(np_.shape(Xnp), th_.shape(Xth))
    assert_tensor_almost_equal(np_.mean(Xnp, axis=1), th_.mean(Xth, axis=1))
