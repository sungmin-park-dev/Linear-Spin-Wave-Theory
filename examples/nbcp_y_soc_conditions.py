"""Check the conditions for extending the classical Y stiffness to SOC.

Use a small-amplitude Fourier perturbation about a fixed global angle, not a
finite uniform spiral. Classical pinning is zero; no quantum gap is inserted.
The signed k coordinate is the one passed to the current Hamiltonian builder.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code-space'), str(ROOT)]

import numpy as np

from examples.nbcp_y_stiffness import S, J, JZ, AREA, DELTAS, PAIRS, GAUGE, POISSON, METRIC, frame, rz
from model.nbcp import make_nn_exchange_matrices, three_msl
from lswt.methods.spin_wave.hamiltonian import LSWTHamiltonian

OUT = ROOT/'data-space/verification/260917-y-soc-conditions'
OMEGA = np.linalg.inv(POISSON)
FIELD = .2


def background(pd, gamma, phi):
    h, c, normals, polar, azimuth = frame(FIELD)
    rotation = rz(phi)
    normals, polar, azimuth = [vectors @ rotation.T for vectors in [normals, polar, azimuth]]
    basis = np.stack([polar, azimuth], axis=-1)
    cfg = dict(Jxy=J, Jz=JZ, JPD=pd, JGamma=gamma, h=(0., 0., h))
    exchanges = make_nn_exchange_matrices(cfg)
    theta = [np.arccos(c), -np.arccos(c), np.pi]
    angles = np.column_stack([theta, np.full(3, phi)]).ravel()
    system = three_msl(cfg, angles, exchanges)
    data = system.to_legacy_dict('Hex_30')
    return h, c, normals, basis, exchanges, angles, data, system


def kernel(q, bg, derivative=()):
    h, _, normals, basis, exchanges, *_ = bg
    matrix = np.zeros((6, 6), complex)
    if not derivative:
        for i in range(3):
            matrix[i, i] = matrix[i+3, i+3] = h*S*normals[i, 2]
    for i, j in PAIRS:
        ii, jj = [i, i+3], [j, j+3]
        for d, exchange in zip(DELTAS, exchanges):
            if not derivative:
                longitudinal = S*S*normals[i] @ exchange @ normals[j]
                matrix[ii, ii] -= longitudinal
                matrix[jj, jj] -= longitudinal
            phase = np.exp(-1j*np.asarray(q)@d)
            for axis in derivative:
                phase *= -1j*d[axis]
            block = S*S*(basis[i].T @ exchange @ basis[j])*phase
            matrix[np.ix_(ii, jj)] += block
            matrix[np.ix_(jj, ii)] += block.conj().T
    return matrix


def reduction(bg):
    s = np.sqrt(1-bg[1]**2)
    g = np.array([0., 0., 0., s, -s, 0.])
    K0 = kernel(np.zeros(2), bg)
    assert np.max(np.abs(K0@g)) < 1e-14
    H = GAUGE.T @ K0 @ GAUGE
    assert np.linalg.eigvalsh(H).min() > 0
    L = np.column_stack([-1j*GAUGE.T @ kernel(np.zeros(2), bg, (i,)) @ g for i in range(2)])
    assert np.max(abs(L.imag)) < 1e-14
    L = L.real
    D = np.array([[np.real(g @ kernel(np.zeros(2), bg, (i, j)) @ g)/2 for j in range(2)] for i in range(2)])
    subtract = np.real(L.T @ np.linalg.solve(H, L))
    assert np.linalg.eigvalsh(subtract).min() > -1e-14
    C = D-subtract
    b = GAUGE.T @ OMEGA @ g
    chi = np.real(b @ np.linalg.solve(H, b))
    drift = np.real(b @ np.linalg.solve(H, L))/chi
    assert abs(chi/3-2/(9*(J+JZ))) < 1e-12
    assert np.linalg.eigvalsh(C).min() > 0
    return g, H, L, D, C, chi, drift


def static_kernel(q, bg, g):
    K = kernel(q, bg)
    z = -np.linalg.solve(GAUGE.T@K@GAUGE, GAUGE.T@K@g)
    vector = g+GAUGE@z
    value = float(np.real(vector.conj()@K@vector))
    assert np.max(abs(GAUGE.T@K@vector)) < 1e-13
    return value, vector


def positive_eigenvalues(matrix):
    values = np.linalg.eigvals(matrix)
    assert np.max(abs(values.imag)) < 1e-10
    values = np.sort(values.real[values.real>0])
    assert len(values)==3
    return values


def geometry_audit(bg):
    data, system = bg[6], bg[7]
    lattice = np.asarray(system.lattice_vectors)
    rows=[]
    for link in data['Couplings']:
        ri=np.array(data['Spin info'][link['SpinI']]['Position'])
        rj=np.array(data['Spin info'][link['SpinJ']]['Position'])
        d=np.array(link['Displacement'])
        f=(d-(rj-ri)) @ np.linalg.inv(lattice)
        r=(d-(ri-rj)) @ np.linalg.inv(lattice)
        rows.append({'pair':link['SpinI']+link['SpinJ'], 'stored_vector':d.tolist(),
                     'forward_translation_integer_residual':float(np.max(abs(f-np.rint(f)))),
                     'reverse_translation_integer_residual':float(np.max(abs(r-np.rint(r))))})
    return rows


def torus_curvature(bg, g, length=24):
    """Independent finite-amplitude energy check with code-consistent embedding.

    Color=(2-n1-2*n2) mod 3 matches the actual A/B/C basis positions. Each
    stored d joins i to j at r_j=r_i-d, contrary to the API's forward wording.
    """
    primitive=np.array([[1.,0.],[.5,np.sqrt(3)/2]])
    coords=np.array([(u,v) for u in range(length) for v in range(length)])
    positions=coords @ primitive
    colors=(2-coords[:,0]-2*coords[:,1]) % 3
    reciprocal=2*np.pi*np.linalg.inv(primitive).T
    q=reciprocal[0]/length
    kappa, vector=static_kernel(q,bg,g)
    wave=np.exp(1j*(positions@q))
    x=np.real(wave*vector[colors]);y=np.real(wave*vector[colors+3])
    normals=bg[2][colors];basis=bg[3][colors]
    neighbor_offsets=[(-1,0),(1,-1),(0,1)]
    def energy(amplitude):
        n=normals*np.sqrt(1-amplitude**2*(x*x+y*y))[:,None]
        n+=amplitude*(basis[:,:,0]*x[:,None]+basis[:,:,1]*y[:,None])
        total=-bg[0]*S*n[:,2].sum()
        for offset, exchange in zip(neighbor_offsets,bg[4]):
            neighbors=(coords+offset) % length
            indices=neighbors[:,0]*length+neighbors[:,1]
            assert np.all(colors[indices]==(colors+1)%3)
            total+=S*S*np.einsum('ni,ij,nj->',n,exchange,n[indices])
        return float(total/len(coords))
    e0=energy(0.)
    checks=[]
    for amplitude in [.002,.001]:
        estimate=6*(energy(amplitude)+energy(-amplitude)-2*e0)/amplitude**2
        error=abs(estimate/kappa-1)
        assert error<2e-4,(estimate,kappa,error)
        checks.append({'amplitude':amplitude,'kappa_per_cell_meV':estimate,'relative_error':error})
    return {'linear_size':length,'spins':length**2,'momentum':q.tolist(),
            'exact_quadratic_kernel_meV_per_cell':kappa,'checks':checks}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    symbolic=json.loads((OUT/'symbolic-soc-check.json').read_text())
    assert all(symbolic['checks'].values())
    assert symbolic['source_sha256']==hashlib.sha256((ROOT/'examples/nbcp_y_soc_conditions.wl').read_bytes()).hexdigest()
    baseline=kernel(np.zeros(2),background(0,0,0))
    cases=[];spectrum_error=0.
    for pd,gamma in [(0.,0.),(.005,0.),(0.,.005),(.005,.005)]:
        for phi in [0.,.173,np.pi/6]:
            bg=background(pd,gamma,phi)
            g,H,L,D,C,chi,drift=reduction(bg)
            assert np.max(abs(kernel(np.zeros(2),bg)-baseline))<1e-14
            ham=LSWTHamiltonian(bg[6]['Spin info'],bg[6]['Couplings'])
            samples=[]
            for angle in [0.,np.pi/4,np.pi/2]:
                direction=np.array([np.cos(angle),np.sin(angle)])
                rho=float(direction@C@direction/(3*AREA))
                for step in [.03,.01,.003,.001]:
                    q=step*direction
                    eps=[]
                    for sign in [1,-1]:
                        mats,torque=ham.Quadratic_Bose_Hamiltonian((sign*q)[None,:],angles=bg[5])
                        assert max(abs(v) for v in torque.values())<1e-12
                        production=positive_eigenvalues(METRIC@mats[0])
                        independent=positive_eigenvalues(1j*POISSON@kernel(sign*q,bg))
                        spectrum_error=max(spectrum_error,float(np.max(abs(production-independent))))
                        eps.append(float(production[0]))
                    predictions=[float(-sign*drift@q+np.sqrt((drift@q)**2+q@C@q/chi)) for sign in [1,-1]]
                    static,_=static_kernel(q,bg,g)
                    rho_static=static/(3*AREA*step**2)
                    rho_product=(chi/3)*eps[0]*eps[1]/(AREA*step**2)
                    samples.append({'direction_rad':angle,'ka':step,'rho_predicted_meV':rho,
                                    'rho_static_meV':rho_static,'rho_dispersion_product_meV':rho_product,
                                    'positive_energy_pm_k_meV':eps,'predicted_energy_pm_k_meV':predictions,
                                    'single_positive_k_naive_rho_meV':(chi/3)*(eps[0]/step)**2/AREA,
                                    'static_relative_error':abs(rho_static/rho-1),
                                    'product_relative_error':abs(rho_product/rho-1),
                                    'energy_relative_error':max(abs(np.array(eps)/predictions-1))})
            finest=[r for r in samples if r['ka']==.001]
            for kind in ['static_relative_error','product_relative_error','energy_relative_error']:
                assert max(r[kind] for r in finest)<1e-5,(pd,gamma,phi,kind,finest)
            case={'JPD_meV':pd,'JGamma_meV':gamma,'phi0_rad':phi,'rho_tensor_meV':(C/(3*AREA)).tolist(),
                  'direct_D_meV_a_squared':D.tolist(),'internal_relaxation_subtraction_meV_a_squared':(D-C).tolist(),
                  'drift_meV_a':drift.tolist(),'chi_per_spin_meV_inverse':chi/3,
                  'hard_eigenvalues_per_cell_meV':np.linalg.eigvalsh(H).tolist(),
                  'gradient_hard_source_norm':float(np.linalg.norm(L)), 'samples':samples}
            if phi==0.:
                case['torus_check']=torus_curvature(bg,g)
            cases.append(case)
    assert spectrum_error<1e-10
    audit=geometry_audit(background(0,.005,0))
    assert max(r['reverse_translation_integer_residual'] for r in audit)<1e-14
    assert min(r['forward_translation_integer_residual'] for r in audit)> .1
    inputs=[Path(__file__),ROOT/'examples/nbcp_y_soc_conditions.wl',ROOT/'examples/nbcp_y_stiffness.py',
            ROOT/'examples/nbcp_ground_state.py',
            ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
            ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py',
            ROOT/'code-space/lswt/system/spin_system.py',
            ROOT/'code-space/lswt/methods/spin_wave/hamiltonian.py']
    report={'scope':'Conditions and local classical harmonic benchmark for Y with SOC; not a quantum-corrected or thermal stiffness',
            'B_T':FIELD,'parameters':{'J_meV':J,'Jz_meV':JZ,'S':S,'a':1.,'area_per_spin':AREA},
            'symbolic_checks':symbolic['checks'],'cases':cases,
            'independent_production_spectrum_max_difference_meV':spectrum_error,
            'bond_orientation_audit':audit,
            'orientation_status':'Stored vectors match r_i-r_j modulo magnetic translations, while SpinSystem API documents r_j-r_i. No production change made; odd-in-k signs quoted in current code k coordinates.',
            'limitations':['Three selected classical angles, not quantum-selected minima.',
                           'Positive hard block and local gradient tensor checked at the sampled points only.',
                           'No full Brillouin-zone stability survey or global phase comparison.',
                           'No zero-point stiffness correction, quantum pinning insertion or finite-temperature calculation.'],
            'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT/'soc-conditions-check.json').write_text(json.dumps(report,indent=2)+'\n')
    compact=[{k:r[k] for k in ['JPD_meV','JGamma_meV','rho_tensor_meV','drift_meV_a']} for r in cases if r['phi0_rad']==0.]
    print(json.dumps({'cases_phi0':compact,'spectral_max_error_meV':spectrum_error,
                      'finest_static_max_relative_error':max(s['static_relative_error'] for c in cases for s in c['samples'] if s['ka']==.001),
                      'finest_product_max_relative_error':max(s['product_relative_error'] for c in cases for s in c['samples'] if s['ka']==.001),
                      'finest_energy_max_relative_error':max(s['energy_relative_error'] for c in cases for s in c['samples'] if s['ka']==.001),
                      'geometry_forward_residual':audit[0]['forward_translation_integer_residual']},indent=2))


if __name__=='__main__':
    main()
