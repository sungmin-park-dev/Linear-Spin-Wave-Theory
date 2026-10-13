import matplotlib
matplotlib.use('TkAgg')

from modules.PhaseDiagram.phase_diagram import PhaseDiagramCalculator
import modules.PhaseDiagram.visualization as vis


MAGNETIC_FIELD_OFFSET = 1e-7 #0.0000001

"""
Basic setting
- nbcp configuration
- optimization setting
- angle setting
- perturbation choose
"""

nbcp_config = {"S": 1/2,
               "Jxy": 0.076,
               "Jz": 0.125,
               "JGamma": 0.0,
               "JPD": 0.0,
            #    "Kxy": 0.0,
            #    "Kz": 0.0,
            #    "KPD": 0.00,
            #    "KGamma": 0.00,
               "h": (0.0, 0.0, 0)}

opt_method = "MAGSWT" # Choose either "classical" or "MAGSWT" or "classical+quantum"
real_time_plot = True
len_k_pts = 15                    # number of k points in the BZ

phi = 0
angles_setting = {"One MSL": (None, phi),
                  "Two MSL": (None, None, None, None),
                  "Three MSL": (None, phi, None, phi, None , phi),
                  "Four MSL": (None, None, None, None, None, None, None, None),}

MAGNETIC_FIELD_OFFSET = 1e-7 #0.0000001
x_start = 0.00 + MAGNETIC_FIELD_OFFSET
x_end   = 0.55
x_step  = 0.2
# x_step  = 0.0025

y_start = 0
y_end   = 0.15
y_step  = 0.15
# y_step  = 0.0025

x_type = "h"
x_range = (x_start, x_end + x_step/2, x_step)

y_type = "JGamma" # Choose either 'JPD' or 'JGamma'.
y_range = (y_start, y_end + y_step/2, y_step)




calculator = PhaseDiagramCalculator(nbcp_config, 
                                    scanning_phase = {x_type: x_range, 
                                                      y_type: y_range},
                                    opt_method = opt_method,
                                    real_time_plot = real_time_plot,
                                    display_method = opt_method,
                                    N = len_k_pts)
    
calculator.calculate_phase_diagram(angles_setting = angles_setting)    



vis.plot_final_results(calculator.output_dir, x_type, y_type, x_range, y_range, show_plot = True)
vis.plot_boson_number(calculator.output_dir, x_type, y_type, x_range, y_range, show_plot = True)
