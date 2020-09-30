# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Adapted from Sklearn (Pedregosa et al 2011):
# github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/_ridge.py
#
# Original Authors: Mathieu Blondel <mathieu@mblondel.org>
#                   Reuben Fletcher-Costin <reuben.fletchercostin@gmail.com>
#                   Fabian Pedregosa <fabian@fseoane.net>
#                   Michael Eickenberg <michael.eickenberg@nsup.org>
#
# License: BSD 3 clause

import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .numtorch import NumTorch, assert_tensor_almost_equal


def correlate(X, Y):
    module = "torch" if isinstance(X, torch.Tensor) else "numpy"
    device = X.device if module == "torch" else None
    npt = NumTorch(module, device=device)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    out = npt.zeros(max([Y.shape[1], X.shape[1]]))
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X ** 2).sum(0) ** 0.5
    SY2 = (Y ** 2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    valid = (SX2 != 0) & (SY2 != 0)
    out[valid] = SXY[valid] / (SX2[valid] * SY2[valid])
    return out


def ridge_cv(X, Y, alphas, independent_alphas=False, Uv=None, scoring="mse"):
    """Similar to sklearn RidgeCV
        - compatible with both numpy and pytorch arrays/tensors
        - optimize a different alpha for each column of Y
        - can use a pretrained svd

    Parameters
    ----------
        X : array | tensor, shape(n_samples, n_features)
            The feature data
        Y: array | tensor, shape(n_samples, n_targets)
            The target data
        alphas : float | array, shape(n_alphas)
            The regularizations coefficients
        independent_alphas : bool
            If True, each target dimension is optimized with a different alpha
        Uv : (array | tensor, array | tensor)
            The result of the SVD, in case it is precomputed.
        scoring : str | callable
            the scoring function used to select alpha. Can be 'mse',
            'correlate' or any function scoring multiple dimensions

    Returns
    -------
        coefs : array | tensor, shape(n_features, n_targets)
            The beta coefficients of the best alpha(s).
        best_alphas : array, shape(n_targets) | float
            The optimal regularizaiton parameter(s).
        Y_hat : array | tensor, shape(n_samples, n_features)
            The leave-one-out predictions.
    """
    assert type(X) == type(Y)
    assert len(X) == len(Y)

    # Prepare function depending on whether data is numpy or torch
    module = "torch" if isinstance(X, torch.Tensor) else "numpy"
    device = X.device if module == "torch" else None
    npt = NumTorch(module, device=device)

    if isinstance(alphas, (float, int)):
        alphas = npt.array(
            [
                alphas,
            ],
            dtype=X.dtype,
        )
    if len(Y.shape) == 1:
        Y = Y[:, None]
    n, n_x = X.shape
    n, n_y = Y.shape

    # Decompose X
    if Uv is None:
        U, s, _ = npt.linealg.svd(X, full_matrices=False)
        v = s ** 2
    else:
        U, v = Uv
    UY = U.transpose(1, 0) @ Y

    # For each alpha, solve leave-one-out error coefs
    cv_duals = npt.zeros((len(alphas), n, n_y), dtype=X.dtype)
    cv_errors = npt.zeros((len(alphas), n, n_y), dtype=X.dtype)

    for alpha_idx, alpha in enumerate(alphas):

        # Solve
        w = ((v + alpha) ** -1) - alpha ** -1
        c = U @ npt.diag(w) @ UY + alpha ** -1 * Y
        cv_duals[alpha_idx] = c

        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        G_diag = (w * U ** 2).sum(-1) + alpha ** -1
        error = c / G_diag[:, None]
        cv_errors[alpha_idx] = error

    # identify best alpha for each column of Y independently
    if independent_alphas:

        if scoring == "mse":
            score = -(cv_errors ** 2).mean(1)
        else:
            if scoring == "correlation":
                scoring = correlate

            score = np.asarray(
                [
                    scoring(Y, Y - cv_errors[alpha_idx])
                    for alpha_idx in range(len(alphas))
                ]
            )

        best_alphas = score.argmax(0)

        duals = npt.zeros((n, n_y), dtype=X.dtype)
        errors = npt.zeros((n, n_y), dtype=X.dtype)

        for i, b in enumerate(best_alphas):
            duals[:, i] = cv_duals[b, :, i]
            errors[:, i] = cv_errors[b, :, i]

    else:
        errors = cv_errors.reshape(len(alphas), -1)
        best_alphas = (errors ** 2).mean(1).argmin(0)
        duals = cv_duals[best_alphas]
        errors = cv_errors[best_alphas]

    coefs = duals.transpose(1, 0) @ X
    Y_hat = Y - errors
    return coefs, best_alphas, Y_hat


class RidgeCV(RegressorMixin, BaseEstimator):
    """Similar to sklearn RidgeCV
        - compatible with both numpy and pytorch arrays/tensors
        - optimize a different alpha for each column of Y
        - can use a pretrained svd

    Parameters
    ----------
        alphas : float | array, shape(n_alphas)
            The regularizations coefficients
        fit_intercept : bool
        normalize : bool
        independent_alphas : bool
            If True, each target dimension is optimized with a different alpha
        pretrain: bool
            When X does not change across successive fits, pretraining can
            accelerate fitting.

    Attributes
    ----------
        self.coefs_ : array | tensor, shape(n_features, n_targets)
        self.alphas_ : array, shape(n_targets) | float
        self.intercept_ : array, shape(n_features, ) | 0.
        self.Uv_ : tuple, in case of pretraining
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        normalize=False,
        independent_alphas=False,
        pretrain=False,
        scoring="mse",
    ):
        self.alphas = alphas
        self.independent_alphas = independent_alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.pretrain = pretrain
        self.scoring = scoring

    def _preprocess(self, X, Y):
        assert len(Y.shape) > 1

        # Prepare function depending on whether data is numpy or torch
        module = "torch" if isinstance(X, torch.Tensor) else "numpy"
        device = X.device if module == "torch" else None
        npt = NumTorch(module, device=device)

        # preprocess
        if self.fit_intercept:
            X_offset = X.mean(0)
            X = X - X_offset
            if self.normalize:
                X_scale = (X ** 2).sum(0) ** 0.5
                X = X / X_scale[None, :]
            else:
                X_scale = npt.ones(npt.shape(X)[1])
            Y_offset = npt.mean(Y, 0)
            Y = Y - Y_offset
        else:
            X_offset = npt.zeros(npt.shape(X)[1], dtype=X.dtype)
            X_scale = npt.ones(npt.shape(X)[1])
            Y_offset = npt.zeros(npt.shape(Y)[1], dtype=X.dtype)

        return X, Y, X_offset, Y_offset, X_scale

    def fit(self, X, Y):
        """Fit cv-optimized ridge regression.

        Parameters
        ----------
            X : array | tensor, shape(n_samples, n_features)
                The feature data
            Y: array | tensor, shape(n_samples, n_targets)
                The target data
        Returns
        -------
            self : the model
        """
        # Preprocess intercept
        X, Y, X_offset, Y_offset, X_scale = self._preprocess(X, Y)

        # Compute X svd, only the first time if pretrained
        if not self.pretrain or not hasattr(self, "Uv_"):
            npt = NumTorch("torch" if isinstance(X, torch.Tensor) else "numpy")
            U, s, _ = npt.linealg.svd(X, full_matrices=False)
            self.Uv_ = (U, s ** 2)

        # Fit Ridge CV
        self.coef_, self.alphas_, _ = ridge_cv(
            X,
            Y,
            alphas=self.alphas,
            independent_alphas=self.independent_alphas,
            Uv=self.Uv_,
            scoring=self.scoring,
        )

        self.coef_ = self.coef_

        # postprocess intercept
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale[None, :]
            Xoff_coef = X_offset[None, :] @ self.coef_.transpose(1, 0)
            self.intercept_ = Y_offset - Xoff_coef
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        assert hasattr(self, "coef_")
        return X @ self.coef_.transpose(1, 0) + self.intercept_


if __name__ == "__main__":
    # Unit tests

    from sklearn.linear_model import RidgeCV as SkRidgeCV
    from sklearn.preprocessing import scale

    def make_data(module="numpy"):
        n, dx, dy = 100, 20, 10
        np.random.seed(0)
        X = np.random.randn(n, dx)
        W = np.random.randn(dx, dy)
        Y = X @ W + np.random.randn(n, dy)
        X, Y = scale(X), scale(Y)
        if module == "torch":
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
            W = torch.from_numpy(W)
        return X, Y, W

    def test_numpy_identical_to_torch():
        X, Y, W = make_data("numpy")
        np_ridge = RidgeCV().fit(X, Y).predict(X)
        X, Y, W = make_data("torch")
        th_ridge = RidgeCV().fit(X, Y).predict(X)
        assert_tensor_almost_equal(np_ridge, th_ridge)

    def test_pretrain():
        X, Y, W = make_data("torch")
        Yhat = RidgeCV(pretrain=False).fit(X, Y).predict(X)
        Yhat_pretrain = RidgeCV(pretrain=True).fit(X, Y).fit(X, Y).predict(X)
        assert_tensor_almost_equal(Yhat_pretrain, Yhat)

    def test_independent_alphas():
        X, Y, W = make_data("torch")
        ridge = RidgeCV(independent_alphas=True).fit(X, Y)
        assert len(ridge.alphas_) == len(Y.T)

        ridge = RidgeCV(independent_alphas=False).fit(X, Y)
        assert len(ridge.alphas_.size()) == 0

    def test_identical_to_scikit():
        X, Y, W = make_data("numpy")
        alphas = [0.1, 1.0, 10.0]
        for fit_intercept in (False, True):
            for normalize in (False, True):
                params = dict(
                    alphas=alphas, fit_intercept=fit_intercept, normalize=normalize
                )
                sk = SkRidgeCV(gcv_mode="svd", **params).fit(X, Y)
                our = RidgeCV(**params).fit(X, Y)
                assert_tensor_almost_equal(sk.coef_, our.coef_)
                assert_tensor_almost_equal(sk.predict(X), our.predict(X))

    def test_scoring():
        X, Y, W = make_data("torch")
        RidgeCV(scoring="correlate").fit(X, Y)
        RidgeCV(scoring="mse").fit(X, Y)
        RidgeCV(scoring=lambda t, p: -np.mean((t - p) ** 2, 0)).fit(X, Y)

    # run tests
    test_numpy_identical_to_torch()
    test_pretrain()
    test_independent_alphas()
    test_identical_to_scikit()
    test_scoring()
