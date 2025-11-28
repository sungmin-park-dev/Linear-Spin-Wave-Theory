import numpy as np
from typing import Union, Optional
"""
Magnon Kernel Functions
----------------------
이 모듈은 스핀파 계산을 위한 다양한 매그논 커널 함수들을 포함합니다.

공통 매개변수:
--------------
E_list : np.ndarray
    에너지 값, 길이는 2*Ns이며 첫 Ns개는 양의 에너지, 나머지 Ns개는 음의 에너지 모드
Temperature : float or int, default=DEFAULT_TEMPERATURE
    온도 (Kelvin). 0이면 기저 상태 계산, 양수만 허용됨
static_corr : np.ndarray, optional
    이미 계산된 정적 상관관계, None인 경우 내부적으로 계산
Ns : int or None, default=None
    서브래티스 수, None인 경우 E_list 길이의 절반으로 설정
eta : float, default=DEFAULT_ETA
    감쇠 계수 (로렌츠/스펙트럴 함수용)
omega : float, default=DEFAULT_OMEGA
    주파수 (로렌츠/스펙트럴 함수용)
time : float, default=DEFAULT_TIME
    시간 (실시간 커널용)

참고 사항:
---------
- 온도가 0인 경우 특별한 처리를 통해 기저 상태의 매그논 분포를 계산합니다.
- 매우 큰 beta*E 값(>BETA_E_THRESHOLD)은 오버플로우를 방지하기 위해 0으로 계산합니다.
- 매우 작은 beta*E 값(<BETA_E_SMALL)은 근사식을 사용하여 1/(beta*E)로 계산합니다.
"""

from modules.constants import K_BOLTZMANN_meV, H_BAR_meV


# 기본 파라미터 상수
DEFAULT_TEMPERATURE = 0       # 기본 온도 (K)
DEFAULT_TIME = 0              # 기본 시간 (무차원)
DEFAULT_OMEGA = 0             # 기본 주파수 (eV)
DEFAULT_ETA = 1e-3            # 기본 감쇠 계수 (eV)

# 계산 임계값 상수
BETA_E_THRESHOLD = 700        # exp() 계산 임계값
BETA_E_SMALL = 1e-10          # 근사식 적용 임계값


def compute_bose_einstein_distribution(E_list: np.ndarray,
                                       Temperature: Union[float, int] = DEFAULT_TEMPERATURE):
    
    beta_E = E_list/(K_BOLTZMANN_meV*Temperature)
    BE_distrbution = np.zeros_like(E_list, dtype=float)
        
    # β*E가 임계값보다 작은 경우에만 계산 수행
    mask_small = beta_E < BETA_E_SMALL
    mask = (beta_E < BETA_E_THRESHOLD) & (~mask_small)
        
    if np.any(mask):
        exp_be = np.exp(beta_E[mask])
        BE_distrbution[mask] = 1.0 / (exp_be - 1.0)
                
    if np.any(mask_small):
        # 매우 작은 x에 대해 1/(exp(x)-1) ≈ 1/x
        BE_distrbution[mask_small] = 1.0 / beta_E[mask_small]
        
    return BE_distrbution


def compute_static_magnon_kernel(E_list: np.ndarray, 
                                 Temperature: Union[float, int] = DEFAULT_TEMPERATURE, 
                                 Ns: Optional[int] = None) -> np.ndarray:
    """정적 매그논 상관관계를 Bose-Einstein 분포를 사용하여 계산합니다."""
    Ns = int(len(E_list)//2) if Ns is None else Ns
    
    normal_ordering = np.hstack([np.ones(Ns), np.zeros(Ns)])
    
    if Temperature == 0:
        return normal_ordering
    
    elif Temperature > 0:
        return normal_ordering + compute_bose_einstein_distribution(E_list, Temperature)

    else:
        raise ValueError("Temperature must be non-negative")


def compute_real_time_kernel(E_list: np.ndarray, 
                             time: Union[float, int] = DEFAULT_TIME, 
                             Temperature: Union[float, int] = DEFAULT_TEMPERATURE, 
                             static_corr: Optional[np.ndarray] = None, 
                             Ns: Optional[int] = None) -> np.ndarray:
    """시간 발전에 따른 매그논 상관관계를 계산합니다."""
    Ns = int(len(E_list)//2) if Ns is None else Ns
    
    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)
    
    if time == 0:
        return static_corr
    
    elif isinstance(time, (float, int, np.number)):
        magnon_energy = np.hstack([E_list[:Ns], -E_list[Ns:]])
        exp_miEt = np.exp(-1j * time * magnon_energy)
        return static_corr * exp_miEt
    
    else:
        raise ValueError(f"Time should be a number, received {time}")


def g_propagator(E_array: np.ndarray, 
                 omega: Union[float, int], 
                 eta: Union[float, int]) -> np.ndarray:
    """그린 함수 전파자를 계산합니다."""
    return 1/(eta + 1j * (E_array - omega))


def compute_lorentzian_kernel(E_list: np.ndarray, 
                             omega: Union[float, int] = DEFAULT_OMEGA,
                             eta: Union[float, int] = DEFAULT_ETA, 
                             Temperature: Union[float, int] = DEFAULT_TEMPERATURE, 
                             static_corr: Optional[np.ndarray] = None, 
                             Ns: Optional[int] = None) -> np.ndarray:
    """로렌츠 형태의 매그논 스펙트럼을 계산합니다."""
    Ns = int(len(E_list)//2) if Ns is None else Ns
    
    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)
    
    Epk = E_list[:Ns]
    Emk = E_list[Ns:]
    
    Fpk = 2 * np.real(g_propagator(Epk, omega, eta))
    Fmk = 2 * np.real(g_propagator(Emk, -omega, eta))
    
    kernel_F = np.hstack([Fpk, Fmk])
    
    return static_corr * kernel_F


def compute_spectral_kernel(E_list: np.ndarray, 
                           omega: Union[float, int] = DEFAULT_OMEGA, 
                           eta: Union[float, int] = DEFAULT_ETA, 
                           Temperature: Union[float, int] = DEFAULT_TEMPERATURE, 
                           static_corr: Optional[np.ndarray] = None, 
                           Ns: Optional[int] = None) -> np.ndarray:
    """스펙트럴 함수 형태의 매그논 상관관계를 계산합니다."""
    Ns = int(len(E_list)//2) if Ns is None else Ns
    
    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)
    
    Epk = E_list[:Ns]
    Emk = E_list[Ns:]
    
    Gpk = 1/2j * (g_propagator(Epk, omega, eta) + g_propagator(Epk, -omega, eta))
    Gmk = 1/2j * (g_propagator(Emk, omega, eta) + g_propagator(Emk, -omega, eta))
    
    kernel_G = np.hstack([Gpk, Gmk.conj()])
    return static_corr * kernel_G