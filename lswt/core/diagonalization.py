"""Bogoliubov-de Gennes diagonalization via Colpa's method.

This module provides the :class:`Diagonalizer` class which implements
bosonic Bogoliubov-de Gennes Hamiltonian diagonalization using Colpa's
algorithm, including several regularization strategies for handling
non-positive-definite Hamiltonians.

Regularization modes
--------------------
- **MAGSWT** (k-independent): a uniform on-site shift is applied to all
  k-points so that the Hamiltonian becomes positive definite everywhere.
- **k-dependent**: each k-point receives the minimal on-site shift needed
  to restore positive definiteness.
- **No regularization**: no shift is applied; if Cholesky fails the
  eigenvalues are obtained from a direct (non-Colpa) diagonalization.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Union

from lswt.config import (
    K_BOLTZMANN_MEV,  # noqa: F401 – reserved for future use
    TOLERANCE_DEFAULT, EPSILON_DEFAULT, THRESHOLD_DEFAULT,
)


class Diagonalizer:
    """Bosonic BdG Hamiltonian diagonalizer using Colpa's algorithm.

    All public methods are class- or static-methods so the class can be
    used without instantiation.
    """

    @staticmethod
    def check_hermiticity(matrix, tolerance=TOLERANCE_DEFAULT):
        """Check whether a matrix is Hermitian within a tolerance.

        Parameters
        ----------
        matrix : np.ndarray
            Square matrix to check.
        tolerance : float, optional
            Absolute tolerance for the element-wise comparison
            (default: ``TOLERANCE_DEFAULT``).

        Returns
        -------
        bool
            ``True`` if ``matrix`` is Hermitian within *tolerance*.
        """
        diff = matrix - matrix.conj().T
        return np.allclose(diff, 0, atol=tolerance)

    @staticmethod
    def check_imag(eigenvalues, tolerance=TOLERANCE_DEFAULT):
        """Separate real and imaginary parts of eigenvalues.

        If all imaginary parts are smaller than *tolerance* the imaginary
        component is returned as ``None``.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Array of (possibly complex) eigenvalues.
        tolerance : float, optional
            Threshold below which imaginary parts are considered zero
            (default: ``TOLERANCE_DEFAULT``).

        Returns
        -------
        re_evals : np.ndarray
            Real parts of *eigenvalues*.
        im_evals : np.ndarray or None
            Imaginary parts, or ``None`` if all are negligible.
        """
        re_evals = np.real(eigenvalues)
        im_evals = np.imag(eigenvalues)
        if np.any(np.abs(im_evals) > tolerance):
            return re_evals, im_evals
        else:
            return re_evals, None

    @staticmethod
    def trace_distance(A, B):
        """Compute the trace distance between two matrices.

        Parameters
        ----------
        A : np.ndarray
            First matrix.
        B : np.ndarray
            Second matrix.

        Returns
        -------
        float
            Trace distance :math:`\\mathrm{Tr}\\sqrt{(A-B)^\\dagger(A-B)}`.
        """
        M = A - B
        D, _ = np.linalg.eigh(M.T.conj() @ M)
        return np.sum(np.sqrt(D))

    @staticmethod
    def Colpa(K, J, paraunitary=True):
        """Colpa diagonalization of a bosonic BdG Hamiltonian.

        Given the Cholesky factor *K* of the Hamiltonian and the
        para-metric matrix *J*, compute the bosonic eigenvalues and,
        optionally, the paraunitary transformation.

        Parameters
        ----------
        K : np.ndarray
            Cholesky factor of the Hamiltonian (upper or lower triangular).
        J : np.ndarray
            Diagonal para-metric matrix with +1 / -1 entries.
        paraunitary : bool, optional
            If ``True`` (default), also return the paraunitary matrix *U*.

        Returns
        -------
        JL : np.ndarray
            Bosonic eigenvalues (positive for particles, negative for holes).
        U : np.ndarray, optional
            Paraunitary transformation matrix (only when *paraunitary* is
            ``True``).
        """
        K_d = K.T.conj()
        L, V = np.linalg.eigh(K_d @ J @ K)
        L = L[::-1]
        V = V[:, ::-1]
        JL = L * np.diag(J)
        if paraunitary:
            inv_K_d = np.linalg.inv(K_d)
            U = inv_K_d @ V @ np.diag(np.sqrt(JL))
            return JL, U
        else:
            return JL

    @classmethod
    def JH_method(cls, H, J):
        """Fallback diagonalization via direct eigendecomposition of J*H.

        Used when the Hamiltonian is not positive definite and Colpa's
        method cannot be applied.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian matrix at a single k-point.
        J : np.ndarray
            Diagonal para-metric matrix.

        Returns
        -------
        eigenvalues : np.ndarray
            Sorted eigenvalues (descending).
        None
            Placeholder for eigenvectors (not computed).
        im_part : np.ndarray or None
            Imaginary parts of the eigenvalues, or ``None`` if negligible.
        """
        M = J @ H
        evals, _ = np.linalg.eig(M)
        re_part, im_part = cls.check_imag(evals)
        if im_part is None:
            eigenvalues = re_part
        else:
            eigenvalues = np.where(np.abs(re_part) >= np.abs(im_part), re_part, im_part)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        return eigenvalues, None, im_part

    @staticmethod
    def reg_method(reg_type):
        """Map a regularization identifier to an integer code.

        Parameters
        ----------
        reg_type : int or str or None
            Regularization specification.  Accepted values:

            - ``0``, ``"MAGSWT"``, ``"k-independent"``, or ``None`` → 0
            - ``1`` or ``"k-dependent"`` → 1
            - ``2`` or ``"No"`` → 2

        Returns
        -------
        int
            Integer regularization code (0, 1, or 2).
        """
        if (reg_type == 0) or (reg_type == "MAGSWT") or (reg_type == "k-independent") or (reg_type is None):
            return 0
        elif (reg_type == "k-dependent") or (reg_type == 1):
            return 1
        elif (reg_type == "No") or (reg_type == 2):
            return 2

    @staticmethod
    def get_kpt_key(k_point):
        """Convert a k-point to a hashable tuple key.

        Parameters
        ----------
        k_point : float or array_like
            Single k-point (scalar or vector).

        Returns
        -------
        tuple of float
            Hashable representation of the k-point.
        """
        if np.isscalar(k_point):
            return (float(k_point),)
        else:
            return tuple(float(x) for x in k_point)

    @classmethod
    def prepare_magswt(cls, Hamiltonian, eps=EPSILON_DEFAULT):
        """Determine the uniform on-site regularization value (MAGSWT).

        Scans all k-points to find the most negative eigenvalue and
        returns a shift that makes every Hamiltonian positive definite.

        Parameters
        ----------
        Hamiltonian : np.ndarray
            Array of shape ``(m, N, N)`` containing the Hamiltonian at
            each k-point.
        eps : float, optional
            Small relative padding added to the shift
            (default: ``EPSILON_DEFAULT``).

        Returns
        -------
        float
            Regularization value (always >= 1e-9).
        """
        m, N, _ = Hamiltonian.shape
        lowest_eval = 0
        for j in range(m):
            eval, _ = np.linalg.eigh(Hamiltonian[j])
            lowest_eval = np.minimum(eval.min(), lowest_eval)
        reg_value = np.abs(lowest_eval * (1 + eps))
        return max(reg_value, 1e-9)

    @classmethod
    def Reg_MAGSWT(cls, Hamiltonian, J_mat, reg_MAGSWT):
        """Apply uniform (MAGSWT) regularization and diagonalize.

        Parameters
        ----------
        Hamiltonian : np.ndarray
            Array of shape ``(m, N, N)``.  Modified in-place.
        J_mat : np.ndarray
            Diagonal para-metric matrix.
        reg_MAGSWT : np.ndarray
            Regularization matrix (``mu * I``).

        Returns
        -------
        Hamiltonian : np.ndarray
            Regularized Hamiltonians.
        Bose_E : list of np.ndarray
            Bosonic eigenvalues per k-point.
        Para_T : list of np.ndarray
            Paraunitary matrices per k-point.
        None
            Placeholder for imaginary eigenvalue data.
        """
        Bose_E = []
        Para_T = []
        for i, H_k in enumerate(Hamiltonian):
            Hamiltonian[i] += reg_MAGSWT
            K = np.linalg.cholesky(Hamiltonian[i])
            bose_Ek, para_Tk = cls.Colpa(K, J_mat, paraunitary=True)
            Bose_E.append(bose_Ek)
            Para_T.append(para_Tk)
        return Hamiltonian, Bose_E, Para_T, None

    @staticmethod
    def get_positive_Hk(Ham, eps=EPSILON_DEFAULT, threshold=THRESHOLD_DEFAULT):
        """Attempt Cholesky decomposition, regularizing if needed.

        If the Hamiltonian is not positive definite, a minimal on-site
        shift is applied (when within *threshold*) before retrying.

        Parameters
        ----------
        Ham : np.ndarray
            Hamiltonian matrix at a single k-point.
        eps : float, optional
            Relative padding for the on-site shift
            (default: ``EPSILON_DEFAULT``).
        threshold : float or None or str, optional
            Maximum allowed negative eigenvalue magnitude.  Special
            values: ``0`` disables regularization, ``None`` or
            ``"max"`` always regularizes (default: ``THRESHOLD_DEFAULT``).

        Returns
        -------
        Ham : np.ndarray
            (Possibly shifted) Hamiltonian.
        K : np.ndarray or None
            Cholesky factor, or ``None`` if decomposition failed.
        """
        try:
            K = np.linalg.cholesky(Ham)
            return Ham, K
        except np.linalg.LinAlgError:
            if threshold == 0:
                return Ham, None
            else:
                eval, _ = np.linalg.eigh(Ham)
                min_eval = np.min(eval)
                onsite = np.abs(min_eval)
                if (threshold is None) or (onsite <= threshold) or (threshold == "max"):
                    Ham += onsite * (1 + eps) * np.eye(Ham.shape[0])
                    K = np.linalg.cholesky(Ham)
                    return Ham, K
                else:
                    return Ham, None

    @classmethod
    def Reg_k_dep_onsite(cls, Hamiltonian, J_mat, eps=EPSILON_DEFAULT, threshold=THRESHOLD_DEFAULT):
        """Apply k-dependent on-site regularization and diagonalize.

        Each k-point is independently shifted by the minimum amount
        required to make the Hamiltonian positive definite.

        Parameters
        ----------
        Hamiltonian : np.ndarray
            Array of shape ``(m, N, N)``.  Modified in-place.
        J_mat : np.ndarray
            Diagonal para-metric matrix.
        eps : float, optional
            Relative padding (default: ``EPSILON_DEFAULT``).
        threshold : float or None or str, optional
            Passed to :meth:`get_positive_Hk`
            (default: ``THRESHOLD_DEFAULT``).

        Returns
        -------
        Hamiltonian : np.ndarray
            Regularized Hamiltonians.
        Bose_E : list
            Bosonic eigenvalues per k-point.
        Para_T : list
            Paraunitary matrices (``None`` where Colpa failed).
        Imag_E : list
            Imaginary eigenvalue components (``None`` where Colpa succeeded).
        """
        Bose_E = []
        Para_T = []
        Imag_E = []
        for i, Hk in enumerate(Hamiltonian):
            reg_Hk, K = cls.get_positive_Hk(Hk, eps=eps, threshold=threshold)
            Hamiltonian[i] = reg_Hk
            if K is not None:
                bose_Ek, para_Tk = cls.Colpa(K, J_mat)
                Bose_E.append(bose_Ek)
                Para_T.append(para_Tk)
                Imag_E.append(None)
            else:
                eig_val, _, im_part = cls.JH_method(reg_Hk, J_mat)
                Bose_E.append(eig_val)
                Para_T.append(None)
                Imag_E.append(im_part)
        return Hamiltonian, Bose_E, Para_T, Imag_E

    @classmethod
    def diagonalize_w_reg(cls, H, regularization, paraunitary=True,
                          eps=EPSILON_DEFAULT, threshold=THRESHOLD_DEFAULT):
        """Diagonalize the full set of Hamiltonians with regularization.

        Parameters
        ----------
        H : np.ndarray
            Array of shape ``(m, N, N)`` with the Hamiltonian at each
            k-point.
        regularization : int or str or None
            Regularization mode (see :meth:`reg_method`).
        paraunitary : bool, optional
            Whether to return paraunitary matrices (default: ``True``).
        eps : float, optional
            Relative padding (default: ``EPSILON_DEFAULT``).
        threshold : float or None or str, optional
            Threshold for k-dependent regularization
            (default: ``THRESHOLD_DEFAULT``).

        Returns
        -------
        H : np.ndarray
            Regularized Hamiltonians.
        Bose_E : list
            Bosonic eigenvalues per k-point.
        Para_T : list
            Paraunitary matrices (only when *paraunitary* is ``True``).
        Imag_E : list or None
            Imaginary eigenvalue data (only when *paraunitary* is ``True``).
        mu_magswt : float or None
            Uniform regularization value (only for MAGSWT mode).
        """
        _, N, _ = H.shape
        Ns = int(N // 2)
        J_mat = np.diag(np.hstack([np.ones((Ns)), -np.ones((Ns))]))
        Identity = np.eye(N)
        reg_type = cls.reg_method(regularization)

        if reg_type == 0:
            mu_magswt = cls.prepare_magswt(H, eps=EPSILON_DEFAULT)
            reg_MAGSWT = mu_magswt * Identity
            H, Bose_E, Para_T, Imag_E = cls.Reg_MAGSWT(H, J_mat, reg_MAGSWT)
        elif reg_type == 1:
            mu_magswt = None
            H, Bose_E, Para_T, Imag_E = cls.Reg_k_dep_onsite(H, J_mat, eps=eps, threshold=threshold)
        elif reg_type == 2:
            mu_magswt = None
            H, Bose_E, Para_T, Imag_E = cls.Reg_k_dep_onsite(H, J_mat, eps=eps, threshold=0)

        if paraunitary:
            return H, Bose_E, Para_T, Imag_E, mu_magswt
        else:
            return H, Bose_E, mu_magswt

    @classmethod
    def get_K_data(cls, k_points, Hamiltonian, regularization,
                   k_indices=None, threshold=THRESHOLD_DEFAULT,
                   partial_derivative_Hk=None):
        """Build per-k-point data dictionary after diagonalization.

        Parameters
        ----------
        k_points : array_like
            List/array of k-point coordinates.
        Hamiltonian : np.ndarray
            Array of shape ``(m, N, N)``.
        regularization : int or str or None
            Regularization mode.
        k_indices : list, optional
            Custom keys for the output dictionary.  If ``None``,
            keys are derived from ``k_points`` via :meth:`get_kpt_key`.
        threshold : float or None or str, optional
            Threshold for k-dependent regularization
            (default: ``THRESHOLD_DEFAULT``).
        partial_derivative_Hk : list of np.ndarray or None, optional
            Partial derivatives of the Hamiltonian
            ``[dH/dk_x, dH/dk_y]``, each of shape ``(m, N, N)``.

        Returns
        -------
        k_data : dict
            Dictionary keyed by k-point.  Each value is a list
            ``[H_k_data, Eigen_data, Colpa_k_data]``.
        mu_magswt : float or None
            Uniform regularization value (if applicable).
        """
        k_data = {}
        ham, bose_E, para_T, imag_E, mu_magswt = cls.diagonalize_w_reg(
            Hamiltonian, regularization, paraunitary=True, threshold=threshold)

        for j, k_pt in enumerate(k_points):
            if k_indices is not None:
                k_key = k_indices[j]
            else:
                k_key = cls.get_kpt_key(k_pt)

            if partial_derivative_Hk is None:
                H_k_data = [ham[j]]
            else:
                H_k_data = [ham[j], partial_derivative_Hk[0][j], partial_derivative_Hk[1][j]]

            Eigen_data = [bose_E[j], para_T[j]]

            if para_T is None:
                Colpa_k_data = [False, imag_E[j]]
            else:
                Colpa_k_data = [True, None]

            k_data[k_key] = [H_k_data, Eigen_data, Colpa_k_data]

        return k_data, mu_magswt

    @classmethod
    def get_eigenvalue(cls, Hamiltonian, reg_type="MAGSWT", threshold=THRESHOLD_DEFAULT):
        """Compute only eigenvalues (without paraunitary matrices).

        Parameters
        ----------
        Hamiltonian : np.ndarray
            Array of shape ``(m, N, N)``.
        reg_type : int or str, optional
            Regularization mode (default: ``"MAGSWT"``).
        threshold : float or None or str, optional
            Threshold for k-dependent regularization
            (default: ``THRESHOLD_DEFAULT``).

        Returns
        -------
        Hamiltonian : np.ndarray
            Regularized Hamiltonians.
        Bose_E : list
            Bosonic eigenvalues per k-point.
        mu_magswt : float or None
            Uniform regularization value (if applicable).
        """
        Hamiltonian, Bose_E, mu_magswt = cls.diagonalize_w_reg(
            Hamiltonian, regularization=reg_type, threshold=threshold, paraunitary=False)
        return Hamiltonian, Bose_E, mu_magswt
