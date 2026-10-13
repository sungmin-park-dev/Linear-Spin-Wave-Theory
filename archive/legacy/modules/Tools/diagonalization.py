import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Union

# Constants for numerical stability
TOLERANCE_DEFAULT = 1e-10
EPSILON_DEFAULT = 1e-6
THRESHOLD_DEFAULT = 1e-8


class DIAG:
    @staticmethod
    def check_hermiticity(matrix: np.ndarray, tolerance: float = TOLERANCE_DEFAULT) -> bool:
        diff = matrix - matrix.conj().T
        return np.allclose(diff, 0, atol=tolerance)

    @staticmethod
    def check_imag(eigenvalues: np.ndarray, tolerance: float = TOLERANCE_DEFAULT) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        re_evals = np.real(eigenvalues)
        im_evals = np.imag(eigenvalues)
        
        if np.any(np.abs(im_evals) > tolerance):
            return re_evals, im_evals
        else: 
            return re_evals, None

    @staticmethod
    def trace_distance(A, B):
        M = A - B
        D, _ = np.linalg.eigh(M.T.conj() @ M)
        return np.sum(np.sqrt(D))
    
    @staticmethod
    def Colpa(K: np.ndarray, J: np.ndarray, paraunitary: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        # K should be checked such that H = KK^{\dagger} 
        K_d = K.T.conj()
        L, V = np.linalg.eigh(K_d @ J @ K)        # L V L^{\dagger} = M        
        # Sorting descending order
        L = L[::-1]                     # L = (Ek1, ... EkNs, - E-kNs, ..., -E-k1 ), first half -> +, other -> -
        V = V[:, ::-1]
        
        JL = L * np.diag(J)             # It should be non-negative.
        
        if paraunitary:                  # return paraunitary
            inv_K_d = np.linalg.inv(K_d)
            U = inv_K_d @ V @ np.diag(np.sqrt(JL))
            return JL, U
        else: 
            return JL  # 단일 값만 반환
            
    @classmethod
    def JH_method(cls, H: np.ndarray, J: np.ndarray) -> Tuple[np.ndarray, None, np.ndarray]:
        M = J @ H
        evals, _ = np.linalg.eig(M)
        re_part, im_part = cls.check_imag(evals)

        if im_part is None:
            eigenvalues = re_part  # If no imaginary part, use only real part
        else:
            eigenvalues = np.where(np.abs(re_part) >= np.abs(im_part), re_part, im_part)
            
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        return eigenvalues, None, im_part
    
    @staticmethod
    def reg_method(reg_type):
        if (reg_type == 0) or (reg_type == "MAGSWT") or (reg_type == "k-independent") or (reg_type is None): 
            return 0
        elif (reg_type == "k-dependent") or (reg_type == 1):
            return 1
        elif (reg_type == "No") or (reg_type == 2):
            return 2

    @staticmethod
    def get_kpt_key(k_point):
        if np.isscalar(k_point):
            k_key = (float(k_point),)
            return k_key
        else:
            k_key = tuple(float(x) for x in k_point)
            return k_key

    
    @classmethod
    def prepare_magswt(cls, Hamiltonian, eps = EPSILON_DEFAULT) -> float:
        m, N, _ = Hamiltonian.shape

        lowest_eval = 0
        for j in range(m):
            eval, _ = np.linalg.eigh(Hamiltonian[j])
            lowest_eval = np.minimum(eval.min(), lowest_eval)
        
        
        # 최소 regularization 값을 1e-8으로 설정 (경험적으로 효과적인 값)
        reg_value = np.abs(lowest_eval*(1 + eps))
        
        return max(reg_value, 1e-9)  # 반드시 1e-8 이상의 값 사용
    
    @classmethod
    def Reg_MAGSWT(cls, Hamiltonian: np.ndarray, J_mat: np.ndarray, reg_MAGSWT: np.ndarray) -> Tuple[np.ndarray, List, List, None]:
        """ MAGSWT regularization
        Ham: np.ndarray (m, 2*Ns, 2*Ns), m = number of k-points, Ns = number of spin sublattices
        J_mat: np.ndarray, (2*Ns, 2*Ns),  diagonal matrix (1,1, ..., -1,-1, ...)
        reg_MAGSWT: regularization matrix
        """
        Bose_E = []
        Para_T = []
        for i, H_k in enumerate(Hamiltonian):
            Hamiltonian[i] += reg_MAGSWT
            K = np.linalg.cholesky(Hamiltonian[i])
            bose_Ek, para_Tk = cls.Colpa(K, J_mat, paraunitary = True)    
            Bose_E.append(bose_Ek)
            Para_T.append(para_Tk)
        
        return Hamiltonian, Bose_E, Para_T, None


    @staticmethod
    def get_positive_Hk(Ham: np.ndarray, eps = EPSILON_DEFAULT, threshold = THRESHOLD_DEFAULT) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            K = np.linalg.cholesky(Ham)
            return Ham, K
        except np.linalg.LinAlgError as e:
            print(f"Warning: {str(e)}. Switching to J-matrix method")
            if threshold == 0:              # no regularization
                return Ham, None
            else:
                eval, _ = np.linalg.eigh(Ham)
                min_eval = np.min(eval)     # non-positive eigenvalue, signature from failure of cholesky
                onsite = np.abs(min_eval)
                
                if (threshold is None) or (onsite <= threshold) or (threshold == "max"):
                    Ham += onsite * (1 + eps) * np.eye(Ham.shape[0])
                    K = np.linalg.cholesky(Ham)
                    return Ham, K
                else:
                    return Ham, None
            
            
    @classmethod
    def Reg_k_dep_onsite(cls, Hamiltonian: np.ndarray, J_mat: np.ndarray, eps=EPSILON_DEFAULT, threshold=THRESHOLD_DEFAULT) -> Tuple[np.ndarray, List, List, List]:
        """
        Regularize Hamiltonian matrices using k-dependent onsite method.
        
        Note: This function modifies the input Hamiltonian array directly for memory efficiency.
        
        Args:
            Hamiltonian: Large array of Hamiltonian matrices
            J_mat: J-matrix for the Colpa method
            eps: Small epsilon for numerical stability
            threshold: Threshold for eigenvalue regularization
            
        Returns:
            Tuple containing (modified Hamiltonian, Bose energies, Paraunitary matrices, Imaginary parts)
        """
        
        Bose_E = []
        Para_T = []
        Imag_E = []
        
        for i, Hk in enumerate(Hamiltonian):
            reg_Hk, K = cls.get_positive_Hk(Hk, eps=eps, threshold=threshold)
            # 정규화된 Hamiltonian 저장 (원본 수정)
            Hamiltonian[i] = reg_Hk
            
            if K is not None:   # regularization succeed
                bose_Ek, para_Tk = cls.Colpa(K, J_mat)
                Bose_E.append(bose_Ek)
                Para_T.append(para_Tk)
                Imag_E.append(None)
            else:               # regularization fail
                print(f"WARNING: Cholesky decomposition failed for k-point {i}. Switching to J-matrix method")
                eig_val, _, im_part = cls.JH_method(reg_Hk, J_mat)
                Bose_E.append(eig_val)
                Para_T.append(None)
                Imag_E.append(im_part)
        
        return Hamiltonian, Bose_E, Para_T, Imag_E


    @classmethod
    def diagonalize_w_reg(cls, H: np.ndarray, 
                        regularization: Union[int, str, None], 
                        paraunitary: bool = True,
                        eps: float = EPSILON_DEFAULT,
                        threshold: float = THRESHOLD_DEFAULT) -> Union[Tuple[np.ndarray, List, List, List, float], 
                                                                                         Tuple[np.ndarray, List, float]]:
        """Diagonalizes the input Hamiltonian H over multiple k-points.

        Args:
            H (np.ndarray): Array of shape (m, N, N) containing m Hamiltonians.
            regularization (Union[int, str, None]): Type of regularization method to use.
                Possible values: 0/"MAGSWT"/"k-independent"/None, 1/"k-dependent", 2/"No"
            paraunitary (bool, optional): Whether to compute paraunitary matrices. Defaults to True.
            eps (float, optional): Small value used for numerical stability. Defaults to EPSILON_DEFAULT.
            threshold (Union[float, str, None], optional): Threshold for regularization. Defaults to THRESHOLD_DEFAULT.

        Returns:
            If paraunitary is True:
                Tuple[np.ndarray, List, List, List, float]: Regularized Hamiltonian, eigenvalues, 
                paraunitary matrices, imaginary parts, and regularization strength.
            Else:
                Tuple[np.ndarray, List, float]: Regularized Hamiltonian, eigenvalues, and 
                regularization strength.
        """
        _, N, _ = H.shape
        Ns = int(N//2)
        J_mat = np.diag(np.hstack([np.ones((Ns)), - np.ones((Ns))]))
        Identity = np.eye(N)

        reg_type = cls.reg_method(regularization)
        
        if reg_type == 0:
            mu_magswt = cls.prepare_magswt(H, eps = EPSILON_DEFAULT)
            # print(f"MAGSWT regularization:\n" f"regularization strength: {mu_magswt}")
            reg_MAGSWT = mu_magswt * Identity 
            H, Bose_E, Para_T, Imag_E = cls.Reg_MAGSWT(H, J_mat, reg_MAGSWT)
        
        elif reg_type == 1:
            print(f"k-dependent regularization with threshold: {threshold}")
            mu_magswt = None
            H, Bose_E, Para_T, Imag_E = cls.Reg_k_dep_onsite(H, J_mat, 
                                                    eps = eps,        # strength of  regularization 
                                                    threshold = threshold)  # criteria for eigenvalue
        
        elif reg_type == 2:
            print("No regularization")
            mu_magswt = None
            H, Bose_E, Para_T, Imag_E = cls.Reg_k_dep_onsite(H, J_mat,
                                                             eps = eps,            # strength of  regularization 
                                                             threshold = 0)         # criteria for eigenvalue
        
        if paraunitary: 
            return H, Bose_E, Para_T, Imag_E, mu_magswt
        else: 
            return H, Bose_E, mu_magswt

    @classmethod
    def get_K_data(cls, k_points: np.ndarray, Hamiltonian: np.ndarray,
                   regularization: Union[int, str, None], 
                   k_indices = None,
                   threshold: Union[float, str, None] = THRESHOLD_DEFAULT,
                   partial_derivative_Hk = None):
        """Wrapping data into dictionary"""
        k_data = {}
        
        ham, bose_E, para_T, imag_E, mu_magswt = cls.diagonalize_w_reg(Hamiltonian, regularization,
                                                                       paraunitary = True, 
                                                                       threshold = threshold)
        
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
    def get_eigenvalue(cls, Hamiltonian: np.ndarray, 
                       reg_type: Union[int, str, None] = "MAGSWT", 
                       threshold: float = THRESHOLD_DEFAULT) -> Tuple[np.ndarray, List, float]:
        """
        Method for computing magnon energy
        """
        Hamiltonian, Bose_E, mu_magswt = cls.diagonalize_w_reg(Hamiltonian, 
                                                               regularization = reg_type,
                                                               threshold = threshold, 
                                                               paraunitary = False)
        return Hamiltonian, Bose_E, mu_magswt

if __name__ == "__main__":
    
    # Number of bands
    n = 4 # This will create a 2n x 2n = 4x4 Hamiltonian
    J = np.diag(np.hstack([np.ones(n), - np.ones(n)]))
    
    A = 2*np.random.rand(n, n) # + 1j*np.random.rand(n, n)
    A = (A + A.T.conj())
    B = np.random.rand(n, n) #+ 1j*np.random.rand(n, n)
    
    
    H =  np.block([[        A,      B   ],
                   [B.T.conj(), A.conj().T ]])
    Eval, U = np.linalg.eigh(H)
    H = U @ np.diag(np.abs(Eval)) @ U.T.conj()
    
    # # # zero mode
    # H = np.array([[ 0.259244792794+0.j, -0.004405599086-0.007630721585j, -0.004405598566+0.007630720686j,  0.+0.j,  0.045844400852+0.079404831531j,  0.045844401371-0.079404832431j],
    #               [-0.004405599086+0.007630721585j,  0.100499996831+0.j, -0.027213759028-0.047135613324j,  0.045844400852-0.079404831531j,  0.+0.j,  0.02303624091 +0.039899939792j],
    #               [-0.004405598566-0.007630720686j, -0.027213759028+0.047135613324j,  0.100500001741+0.j,  0.045844401371+0.079404832431j,  0.02303624091 -0.039899939792j,  0.+0.j],
    #               [ 0.+0.j,  0.045844400852+0.079404831531j,  0.045844401371-0.079404832431j,  0.259244792794+0.j, -0.004405599086-0.007630721585j, -0.004405598566+0.007630720686j],
    #               [ 0.045844400852-0.079404831531j,  0.+0.j,  0.02303624091 +0.039899939792j, -0.004405599086+0.007630721585j,  0.100499996831+0.j, -0.027213759028-0.047135613324j],
    #               [ 0.045844401371+0.079404832431j,  0.02303624091 -0.039899939792j,  0.+0.j, -0.004405598566-0.007630720686j, -0.027213759028+0.047135613324j,  0.100500001741+0.j]])
    
    # negative
    #H = np.array([[ 0.259244789604+0.j, 0.019885588104-0.033988428011j, 0.019885588192+0.033988428244j, 0.+0.j, -0.013392765335+0.031645348204j, -0.013392765248-0.031645347971j],[ 0.019885588104+0.033988428011j, 0.100500001966+0.j, 0.010170849225-0.0224802906j, -0.013392765335-0.031645348204j, 0.+0.j, -0.023107504215+0.043153485615j],[ 0.019885588192-0.033988428244j, 0.010170849225+0.0224802906j, 0.100500000647+0.j, -0.013392765248+0.031645347971j, -0.023107504215-0.043153485615j, 0.+0.j],[ 0.+0.j, -0.013392765335+0.031645348204j, -0.013392765248-0.031645347971j, 0.259244789604+0.j, 0.019885588104-0.033988428011j, 0.019885588192+0.033988428244j],[ -0.013392765335-0.031645348204j, 0.+0.j, -0.023107504215+0.043153485615j, 0.019885588104+0.033988428011j, 0.100500001966+0.j, 0.010170849225-0.0224802906j],[ -0.013392765248+0.031645347971j, -0.023107504215-0.043153485615j, 0.+0.j, 0.019885588192-0.033988428244j, 0.010170849225+0.0224802906j, 0.100500000647+0.j]])
    
    # exceptional case: Gamma gapless point
    H = np.array([
        [0.1005 + 0.j, 0.020833767361 + 0.j, 0.016662760417 + 0.j, 0. + 0.j, -0.079666232639 + 0.j, -0.083837239583 - 0.j],
        [0.020833767361 + 0.j, 0.1005 + 0.j, 0.016662760417 + 0.j, -0.079666232639 + 0.j, 0. + 0.j, -0.083837239583 - 0.j],
        [0.016662760417 - 0.j, 0.016662760417 - 0.j, 0.245651041667 + 0.j, -0.083837239583 - 0.j, -0.083837239583 - 0.j, 0. + 0.j],
        [0. + 0.j, -0.079666232639 - 0.j, -0.083837239583 + 0.j, 0.1005 + 0.j, 0.020833767361 + 0.j, 0.016662760417 - 0.j],
        [-0.079666232639 - 0.j, 0. + 0.j, -0.083837239583 + 0.j, 0.020833767361 + 0.j, 0.1005 + 0.j, 0.016662760417 - 0.j],
        [-0.083837239583 + 0.j, -0.083837239583 + 0.j, 0. + 0.j, 0.016662760417 + 0.j, 0.016662760417 + 0.j, 0.245651041667 + 0.j]
    ])

    
    n = len(H)//2
    J = np.diag(np.hstack([np.ones(n), - np.ones(n)]))
    
    # Test single matrix with Colpa method
    print("\nTesting positive definite matrix:")
    eval, _ = np.linalg.eigh(H)
    print(f"minimum eigen value: \n{eval.min()}")
    K = np.linalg.cholesky(H + np.eye(n*2)*1e-12)

    eval, U = DIAG.Colpa(K, J)
    print(f"\nColpa's diagonalization:\n{eval}")
    print(f"Bosonic commutation: \n{np.diag(U @ J @ U.T.conj())}")
    