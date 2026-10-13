import json
import numpy as np
from pathlib import Path
from datetime import datetime

from datetime import datetime
from pathlib import Path

def create_output_directory(scan_types: tuple):
    """Create and return output directory path under 'data/'"""

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    
    dir_name = "phase_diagram"
    
    if scan_types:
        for scan_var in scan_types:
            dir_name += f"_{scan_var}"
    
    dir_name += f"_{timestamp}"

    # 'data/' 경로 하위에 디렉토리 생성
    output_dir = Path("datas") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_configuration(output_dir, config):
    """Save calculation configuration to JSON"""
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)




def save_point_data(output_dir, filename, point_data):
    
    # point_data 출력
    # print(json.dumps(point_data, indent=4))
    
    with open(output_dir / filename, 'w') as f:
        json.dump(point_data, f, indent=4)

    return 

def save_final_results(output_dir, phase_map, fig=None):
    """Save phase map and final plot"""
    # Phase map 저장
    phase_map_path = output_dir / 'phase_map.npy'
    np.save(phase_map_path, phase_map)
    
    # 플롯 저장 (fig이 제공된 경우)
    if fig is not None:
        plot_path = output_dir / 'phase_diagram_final.png'
        fig.savefig(plot_path, bbox_inches='tight', dpi=600)
        return plot_path
    
    return phase_map_path