import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .phase_analyzer import Phase, PHASE_STYLES, get_phase_style, get_phase_name

"""
    # 퇴화 상태 산점도 초기화
    degenerate_scatter = ax.scatter(
        [], [], 
        c='#000000',
        marker='x',
        s=100,
        alpha=0.4,
        label='Degenerate States'
    )
"""

def initialize_plot(x_type, x_range, y_type, y_range):
    """Initialize phase diagram plot with fixed axes"""
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.show(block=False) 
    
    
    # 그리드 설정
    ax.grid(True, linestyle='-', alpha=0.3, which='major', color='gray')
    ax.grid(True, linestyle=':', alpha=0.2, which='minor', color='gray')
    ax.minorticks_on()
    
    # 축 설정
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    new_x_end = x_range[1] - x_range[2] / 2
    new_y_end = y_range[1] - y_range[2] / 2
    new_x_step = (new_x_end - x_range[0]) / 10
    new_y_step = (new_y_end - y_range[0]) / 10
    ax.set_xticks(np.arange(x_range[0], new_x_end, new_x_step))
    ax.set_yticks(np.arange(y_range[0], new_y_end, new_y_step))
    
    # Phase별 산점도 초기화
    scatters = {}
    for phase in Phase:
        style = get_phase_style(phase)
        scatters[phase] = ax.scatter([], [], 
                                     c=style['color'],
                                     marker=style['marker'],
                                     s=style['size'],
                                     edgecolors=style['edgecolor'],
                                     linewidth=style['linewidth'],
                                     label=f"{get_phase_name(phase)} ({phase.value})",
                                     alpha=0.8)
    
    # 라벨 및 제목 설정
    ax.set_xlabel(f'{x_type} (meV)', fontsize=12)
    ax.set_ylabel(f'{y_type} (meV)', fontsize=12)
    ax.set_title(f'Phase Diagram ({x_type} vs {y_type})', fontsize=14)
    
    # 범례 설정
    legend = ax.legend(bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      fontsize=10,
                      frameon=True,
                      title='Phases',
                      title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 배경 저장
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    
    return fig, ax, scatters, background

def update_plot(fig, ax, value_x, value_y, phase, scatters = None, background=None):
    """Update scatter plot with new point"""
    # 배경 복원
    fig.canvas.restore_region(background)
    
    # Phase 산점도 업데이트
    scatter = scatters[phase]
    new_data = np.array([[value_x, value_y]])
    old_data = scatter.get_offsets()
    
    if len(old_data) > 0:
        scatter.set_offsets(np.vstack([old_data, new_data]))
    else:
        scatter.set_offsets(new_data)
        
    # 업데이트된 아티스트 그리기
    for scatter in scatters.values():
        ax.draw_artist(scatter)
        
    # 디스플레이 업데이트
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    plt.pause(0.01)


def plot_final_results(result_dir, x_type, y_type, x_range, y_range, show_plot=True):
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    result_dir = Path(result_dir)
    
    # 설정 로드
    config_path = result_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # JSON 파일에서 데이터 로드 (MAGSWT 결과 사용)
    phase_map_path = result_dir / 'phase_map_magswt.json'
    if not phase_map_path.exists():
        raise FileNotFoundError(f"Phase map file not found at {phase_map_path}")
    
    # JSON 파일을 열고 데이터 로드
    with open(phase_map_path, 'r') as f:
        phase_map_data = json.load(f)
    
    # JSON 데이터를 NumPy 배열로 변환
    phase_map = np.array(phase_map_data)
    
    # Phase diagram 플롯
    plt.figure(figsize=(15, 10))
    
    # Phase별 색상 매핑
    unique_phases = np.unique(phase_map)
        
    nx, ny = phase_map.shape
    x_values = np.arange(*x_range)
    y_values = np.arange(*y_range)

    print(f"nx: {nx}, ny: {ny}")
    print(f"len(x_values): {len(x_values)}, len(y_values): {len(y_values)}")

    if len(x_values) != nx or len(y_values) != ny:
        raise ValueError(f"Mismatch between phase_map shape {phase_map.shape} "
                        f"and x/y value lengths ({len(x_values)}, {len(y_values)})")
    
    # 각 상(phase)별로 산점도 생성
    for phase_idx in unique_phases:
        mask = (phase_map == phase_idx)
        phase = Phase(phase_idx)
        x_coords = []
        y_coords = []
        
        for i in range(nx):      # y 축
            for j in range(ny):  # x 축
                if mask[i, j]:   # mask[i, j] 접근 시 범위 초과 방지
                    x_coords.append(x_values[i])  # x는 j에 해당
                    y_coords.append(y_values[j])  # y는 i에 해당
        
        style = get_phase_style(phase)
        plt.scatter(x_coords, y_coords,
                   c=style['color'],
                   marker=style['marker'],
                   s=style['size'],
                   edgecolors=style['edgecolor'],
                   linewidth=style['linewidth'],
                   label=f"{get_phase_name(phase)} ({phase.value})",
                   alpha=0.8)
    
    # 축 범위와 그리드 설정
    plt.xlim(x_range[0], x_range[1] - x_range[2] / 2)
    plt.ylim(y_range[0], y_range[1] - y_range[2] / 2)
    plt.grid(True, linestyle='-', alpha=0.3, which='major', color='gray')
    plt.grid(True, linestyle=':', alpha=0.2, which='minor', color='gray')
    plt.minorticks_on()

    
    plt.xlabel(f'{x_type} (meV)')
    plt.ylabel(f"{y_type} (meV)")
    plt.title(f"Phase Diagram ({x_type} vs {y_type})")
    
    # 범례 설정
    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left',
                        fontsize=10,
                        frameon=True,
                        title='Phases',
                        title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 결과 저장
    plt.savefig(result_dir / 'phase_diagram_final.png', 
                bbox_inches='tight', dpi=600)
    
    if show_plot:
        plt.show()
    plt.close()


def plot_boson_number(result_dir, x_type, y_type, x_range, y_range, show_plot=True):
    """Create scatter plot of average boson number from stored results"""
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    x_start, x_end, x_step = x_range
    y_start, y_end, y_step = y_range

    result_dir = Path(result_dir)

    # 설정 로드
    config_path = result_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Physical quantities 로드
    phys_quant_path = result_dir / 'physical_quantities_magswt.json'
    if not phys_quant_path.exists():
        raise FileNotFoundError(f"Physical quantities file not found at {phys_quant_path}")
    
    # JSON 파일에서 데이터 로드
    with open(phys_quant_path, 'r') as f:
        phys_quant_data = json.load(f)
    
    # phys_quant_data 크기
    rows = len(phys_quant_data)
    cols = len(phys_quant_data[0]) if rows > 0 else 0

    valid_points = []
    valid_values = []
    invalid_points = []

    # 데이터 순회 및 값 추출
    for i in range(rows):
        for j in range(cols):
            data = phys_quant_data[i][j]
            print(f"phys_quant_data[{i}][{j}]: {data}")
            if data is not None and isinstance(data, dict):
                # x_type, y_type 값 추출
                x = data.get(x_type)
                y = data.get(y_type)
                # physical_quantities에서 boson_number 추출
                phys_quant = data.get('physical_quantities', {})
                boson_number = phys_quant.get('boson_number', {})
                avg_boson_number = boson_number.get('average', None)

                if x is not None and y is not None and avg_boson_number is not None:
                    if avg_boson_number <= 1:
                        valid_points.append((x, y))
                        valid_values.append(avg_boson_number)
                    else:
                        invalid_points.append((x, y))
                else:
                    invalid_points.append((x, y))
                    print(f"Missing data at [{i}, {j}]: x={x}, y={y}, boson_number={boson_number}")
            else:
                print(f"Invalid or None data at [{i}, {j}]: {data}")
                # x, y 값이 없으므로 무효 처리 (필요 시 x_range, y_range 기반으로 처리 가능)
                invalid_points.append((None, None))

    # 플롯
    plt.figure(figsize=(15, 10))

    # 유효한 점들 (색상)
    if valid_points:
        valid_points = np.array([(x, y) for x, y in valid_points if x is not None and y is not None])
        if len(valid_points) > 0:
            scatter = plt.scatter(
                valid_points[:, 0], valid_points[:, 1],
                c=valid_values,
                cmap='viridis',
                s=100,
                alpha=0.8,
                edgecolor='k',
                label='Valid (boson number ≤ 1)'
            )
            plt.colorbar(scatter, label='Average Boson Number (valid only)')

    # 무효한 점들 (빨간 X)
    if invalid_points:
        invalid_points = np.array([(x, y) for x, y in invalid_points if x is not None and y is not None])
        if len(invalid_points) > 0:
            plt.scatter(
                invalid_points[:, 0], invalid_points[:, 1],
                marker='x',
                color='red',
                s=100,
                label='Invalid (boson number > 1 or missing)'
            )

    # 축 설정
    plt.xlim(x_start, x_end)
    plt.ylim(y_start, y_end)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.minorticks_on()

    # 축 눈금
    new_x_end = x_end - x_step / 2
    new_y_end = y_end - y_step / 2
    new_x_step = (new_x_end - x_start) / 10
    new_y_step = (new_y_end - y_start) / 10
    plt.xticks(np.arange(x_start, new_x_end + new_x_step, new_x_step))
    plt.yticks(np.arange(y_start, new_y_end + new_y_step, new_y_step))

    plt.xlabel(f'{x_type} (meV)')
    plt.ylabel(f'{y_type} (meV)')
    plt.title(f'Average Boson Number ({x_type} vs {y_type})')
    plt.legend()
    plt.tight_layout()

    # 저장 및 표시
    plt.savefig(result_dir / 'boson_number_final.png', bbox_inches='tight', dpi=600)
    if show_plot:
        plt.show()
    plt.close()