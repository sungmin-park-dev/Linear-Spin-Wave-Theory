import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing necessary modules
from modules.SpinSystem.nbcp_phases import FIND_TLAF_PHASES

from modules.Plotters.spin_system_visualizer import SpinVisualizer

from modules.LinearSpinWaveTheory.linear_spin_wave_theory import LSWT
from modules.Plotters.updated_band_plotter import Band_Plotter
from modules.Plotters.momentum_space_plotter import MomentumSpacePlotter

from modules.Tools.analysis_tools import Create_Energy_Function as cef
from modules.Tools.analysis_tools import calculate_skyrmion_number



# ==========================================================
nbcp_config = {"Jxy": 0.076,    # nearest neighbor exchange
               "Jz": 0.125,
               "JGamma": 0.1,
               "JPD": 0.00,
               "Kxy": 0.0,      # anisotropy
               "Kz": 0.00,
               "KPD": 0.00,
               "KGamma": 0.0,
               "h": (0.00, 0.00, 0.376418,)} 

# optimization setting
opt_method = "MAGSWT" # Choose either "classical" or "MAGSWT"
N = 20                # number of k points in the BZ, ~N^2
phi = 0
temperature = 0


angles_setting = {"One MSL": (None, phi),
                  "Two MSL": (None, None, None, None),
                  "Three MSL": (None, phi, None, phi, None, phi),
                  "Four MSL": (None, None, None, None, None, None, None, None),}


check_classical_stability = False    # check the optimization results lying on the classical stable point

do_plot_spin_conf = True
do_plot_band = True

do_analysis = True  # Chern, boson numer
bz_type = "Hex_60"  # choose either "simple" (First BZ) "Hex_60, Hex_30, Tetra"
exclude_gamma = False

do_plot_thermodynamics = True
do_plot_boson_number = False    #
do_plot_real_space_corr = False
do_plot_structure_factor = False
do_plot_spectral_along_hsp = False
# ==========================================================



"""
Find nbcp phase
"""
nbcp = FIND_TLAF_PHASES(config = nbcp_config)       # class 정의


# nbcp
nbcp_opt_result, nbcp_cls_result = nbcp.find_tlaf_phase(opt_method = opt_method, 
                                                       angles_setting = angles_setting,
                                                       verbose = True, 
                                                       N = N)   
nbcp.summarize_results(verbose = True)






if check_classical_stability:
    classical_stability = cef.Diff_classical_energy_per_site_func(nbcp_opt_result["spin_sys_data"])
    
    for key, value in classical_stability.items():
        print(f"{key}-derivatives: ")
        for diff_var, diff_value in value.items():
            print(f"{diff_var}: {diff_value}")


if do_plot_spin_conf:
    visualizer = SpinVisualizer()
    
    best_opt = nbcp_opt_result["spin_sys_data"]
    best_cls = nbcp_cls_result["spin_sys_data"]
    
    
    
    fig, (ax_xy, ax_angle) = visualizer.plot_system(best_opt)
    fig.suptitle(f"NBCP spin Configuration", fontsize=16, y=1.05)
    plt.show()
    
    fig, (ax_xy, ax_angle) = visualizer.plot_system(best_cls)
    fig.suptitle(f"NBCP spin Configuration", fontsize=16, y=1.05)
    plt.show()

    sky_number = calculate_skyrmion_number(best_opt)
    print(f"skyrmion\n{sky_number}")
    sky_number = calculate_skyrmion_number(best_cls)
    print(f"skyrmion\n{sky_number}")

if do_plot_band:
    bandplotter = Band_Plotter(nbcp_opt_result["spin_sys_data"], 
                               bz_type = bz_type)
    bandplotter.prepare_HSP_path(reg_type = 0, 
                                 threshold= 1e-8, 
                                 compute_berry_curvature = True,    # False
                                 mag = 200)
    result = bandplotter.plot_band_w_number(set_path_name = "standard path")
    
    dfs = []

    # 자동으로 key를 순회하며 열 이름 생성 및 DataFrame 결합
    # for key in result.keys():
    #     arr = result[key]
    #     if isinstance(arr, np.ndarray) and arr.ndim == 2:
    #         df_part = pd.DataFrame(arr, columns=[f'{key}_{i}' for i in range(arr.shape[1])])
    #         dfs.append(df_part)
    #     else:
    #         print(f"Skip '{key}': not a 2D numpy array.")

    # # 옆으로 붙이기 (열 단위로)
    # df_full = pd.concat(dfs, axis=1)

    # df_full.to_excel("Stripe_YZ_B_03T.xlsx", index=False)

"""
Analysis given state
"""

lswt = LSWT(nbcp_opt_result["spin_sys_data"])
k_data, bz_data, full_k_points = lswt.diagnosing_lswt(bz_type = bz_type, 
                                                      N = 50, 
                                                      regularization = "MAGSWT", 
                                                      temperature=0)
    

if do_analysis:    
    Berry_curv, chern_number, THC = lswt.topo.compute_thermal_Hall(k_data, Temperature =  0.1)
    lswt.topo.plot_berry_curvature(full_k_points, Berry_curv, band_index = 1)
    lswt.topo.plot_berry_curvature(full_k_points, Berry_curv, band_index = 2)
    lswt.topo.plot_berry_curvature(full_k_points, Berry_curv, band_index = 3)
    lswt.topo.plot_berry_curvature(full_k_points, Berry_curv, band_index = 4)
    
    
    E_int = lswt.ther.compute_internal_energy(k_data, Temperature = 0, invalid_exclude = False)
    sublat = lswt.msl_average_boson_number
    total  = lswt.average_boson_number 
    
    labels = ["Total"] + list(sublat.keys())
    data = np.concatenate([[total], [value for value in sublat.values()]])
    # 수정: 초기 플롯 코드 복원, 시각화 개선
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print(f"Error: Invalid plot data: {data}")
    else:
        plt.figure(figsize=(8, 6))
        plt.bar(labels, data, color=['blue', 'green', 'red'], edgecolor='black', width=0.4)
        plt.title('Average Bosonic Occupation Number for Total and Sublattices')
        plt.xlabel('Total and Sublattices')
        plt.ylim((0, np.maximum(0.5, np.round(np.max(data)) + 1) ))
        plt.ylabel('Average Bosonic Occupation Number')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if do_plot_thermodynamics:
    temperatures, results =  lswt.ther.get_thermodynamic_quantities(k_data, Temperature_range = (0.0, 10, 0.2))
    lswt.ther.plot_thermodynamic_quantities(temperatures, results)
    
    # 1. 스칼라 값을 저장할 DataFrame 생성
    df_scalar = pd.DataFrame({
        'Temperature': temperatures,
        'Internal Energy Density': results['Internal Energy Density'],
        'Entropy Density': results['Entropy Density'],
        'Specific Heat Density': results['Specific Heat Density'],
        'Thermal Hall Conductance': results['Thermal Hall Conductance'],
        'Total Boson Number': results['Total Boson Number'],
    })

    # 2. Sublattice Boson Numbers는 다차원 배열이므로 따로 처리
    Ns = results['Sublattice Boson Numbers'].shape[0]
    sublattice_data = {}

    for s in range(Ns):
        sublattice_data[f'Sublattice {s} Boson Number'] = results['Sublattice Boson Numbers'][s]

    df_sublattice = pd.DataFrame(sublattice_data)
    df_sublattice.insert(0, 'Temperature', temperatures)  # 온도 열 추가

    # 3. 두 테이블을 엑셀의 여러 시트로 저장
    with pd.ExcelWriter("thermodynamic_results.xlsx") as writer:
        df_scalar.to_excel(writer, sheet_name='Thermodynamic Scalars', index=False)
        df_sublattice.to_excel(writer, sheet_name='Sublattice Bosons', index=False)

    print("엑셀 파일로 저장 완료: thermodynamic_results.xlsx")


if do_plot_boson_number:
    # # 모멘텀 공간 보존 넘버 시각화
    plotter = MomentumSpacePlotter(lswt, bz_data)
    
    sublattice_boson_numbers, all_boson_numbers, eigenvalues_array, valid_count = lswt.ther.bosonic_momentum_correlation(k_data, Temperature = temperature)
    momentum_space_fig = plotter.plot_number_data(full_k_points, 
                                                  all_boson_numbers, 
                                                  sublattice_boson_numbers, component_names = ["A", "B", "C", "D"])

    plt.tight_layout()
    plt.show()

if do_plot_real_space_corr:
    r_vals, corr_data = lswt.corr.compute_real_space_correlations(k_data,
                                                                  angle_direction = 0, #np.pi/6, 
                                                                  distance_range = (0, 20, 2),
                                                                  time = 0 
                                                                  )
    lswt.corr.plot_real_space_correlation(r_vals, corr_data,
                                          comb_to_plot = [(0, 0), (1,1), (0, 1), (2, 3)],
                                          component='abs',
                                          log_x=False,
                                          log_y=False)


if do_plot_structure_factor:
    plotter = MomentumSpacePlotter(lswt, bz_data)
    
    fig = lswt.corr.plot_correlation_FBZ(k_data, 
                                         func_type="structure factor",
                                         # func_type = "spectral function",
                                         coordinate_type = "cartesian",
                                         classical_contribution = True,
                                         sublattice=None,
                                         Temperature = temperature,
                                         omega = None,
                                         plotter_obj=plotter)

    plt.show()
    

if do_plot_spectral_along_hsp:
    print("\n" + "="*50)
    print("스펙트럴 함수 계산 및 시각화")
    print("="*50)
    
    path_type = 0
    available_path = list(bandplotter.band_paths.keys())[path_type]
    print(f"선택된 경로: {available_path}")

    bandplotter.prepare_dynamic_function(path_name = available_path,  # 실제 사용 가능한 경로 이름 사용
                                         func_type = "spectral",
                                         msl_boson_number = lswt.msl_average_boson_number,
                                         omega_max = 0.3, 
                                         Temperature = 0,
                                         num_omega_points = 61, 
                                         decay_factor = 0.00001,
                                         mag = 50,
                                         classical_contribution = True)
    fig_1x3 = bandplotter.plot_spectral_functions_1x3(
        cmap='coolwarm',
        log_scale=True,
        share_colorbar=False,
        contrast_enhance=0.97  # 상위 5%의 값을 잘라냄 (대비 강화)
    )
    plt.show()

