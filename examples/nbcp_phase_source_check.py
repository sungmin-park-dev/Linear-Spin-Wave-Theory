"""Check equations adopted from the historical NBCP notebook.

This bounded diagnostic verifies local algebra and reference field values.
It does not search for a global phase diagram or run a new spin-wave scan.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data-space/verification/260916-nbcp-integration'


def tensor(phi, j, jz, pd, gamma):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[j + 2 * pd * c, -2 * pd * s, -gamma * s],
                     [-2 * pd * s, j - 2 * pd * c, gamma * c],
                     [-gamma * s, gamma * c, jz]])


def main():
    spin, jz, gz, mu_b = .5, .125, 4.645, .05788381806
    rows = []
    for j in [.075, .076]:
        h = np.array([3 * spin * j,
                      3 * spin * (jz - j / 2 + np.sqrt(jz**2 + jz*j - 7*j*j/4)),
                      3 * spin * (j + 2 * jz)])
        uud = lambda f: np.array([[f/3, j*spin, j*spin],
                                 [j*spin, f/3, j*spin],
                                 [j*spin, j*spin, 2*jz*spin-f/3]])
        polar = np.full((3, 3), j*spin)
        np.fill_diagonal(polar, h[2]/3 - 2*jz*spin)
        rows.append({'J_meV': j, 'h_critical_meV': h.tolist(),
                     'B_critical_T': (h/(gz*mu_b)).tolist(),
                     'UUD_boundary_eigenvalues': [np.linalg.eigvalsh(uud(f)).tolist()
                                                  for f in h[:2]],
                     'UUD_inside_min': float(np.linalg.eigvalsh(uud(np.mean(h[:2])))[0]),
                     'UUD_outside_min': [float(np.linalg.eigvalsh(uud(f))[0])
                                         for f in [h[0]-1e-4, h[1]+1e-4]],
                     'P_boundary_eigenvalues': np.linalg.eigvalsh(polar).tolist()})

    rng = np.random.default_rng(916)
    errors = {'three_sublattice_energy': [], 'stripe_energy': [],
              'V_gradient': [], 'historical_Gamma_to_V_geometry': [],
              'UUD_Cartesian_Hessian': []}
    j, pd, gamma = .075, .006, .009
    bonds = [tensor(p, j, jz, pd, gamma) for p in [0, 2*np.pi/3, -2*np.pi/3]]
    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi, 3)
        phi = rng.uniform(-np.pi, np.pi, 3)
        spins = spin*np.column_stack((np.sin(theta)*np.cos(phi),
                                      np.sin(theta)*np.sin(phi), np.cos(theta)))
        h = rng.uniform(.1, .4)
        pairs = [(0, 1), (1, 2), (2, 0)]
        bond_e = sum(spins[a]@b@spins[c] for a,c in pairs for b in bonds)/3
        bond_e -= h*spins[:, 2].sum()/3
        reduced = sum(j*np.dot(spins[a,:2],spins[c,:2])+jz*spins[a,2]*spins[c,2]
                      for a,c in pairs)-h*spins[:,2].sum()/3
        errors['three_sublattice_energy'].append(abs(bond_e-reduced))

        ta, tb = theta[:2]
        a = spin*np.array([0, np.sin(ta), np.cos(ta)])
        b = spin*np.array([0, np.sin(tb), np.cos(tb)])
        stripe = a@bonds[0]@a+b@bonds[0]@b+2*a@(bonds[1]+bonds[2])@b-h*(a[2]+b[2])
        sa,sb,ca,cb = np.sin(ta),np.sin(tb),np.cos(ta),np.cos(tb)
        stripe_old = spin**2*(j*(sa*sa+sb*sb+4*sa*sb)+jz*(ca*ca+cb*cb+4*ca*cb)
                                 -2*pd*(sa-sb)**2+2*gamma*(ca-cb)*(sa-sb))-h*spin*(ca+cb)
        errors['stripe_energy'].append(abs(stripe-stripe_old))

        def ev(t, v):
            return spin**2*(j*(np.sin(t)**2+2*np.sin(t)*np.sin(v))
                            +jz*(np.cos(t)**2+2*np.cos(t)*np.cos(v))) \
                            -h*spin*(2*np.cos(t)+np.cos(v))/3
        t, v = theta[:2]
        grad = np.array([2*spin**2*(j*np.cos(t)*(np.sin(t)+np.sin(v))
                                    -jz*np.sin(t)*(np.cos(t)+np.cos(v)))+2*h*spin*np.sin(t)/3,
                         spin**2*(2*j*np.sin(t)*np.cos(v)-2*jz*np.cos(t)*np.sin(v))
                                    +h*spin*np.sin(v)/3])
        step=1e-5
        fd = np.array([(ev(t+step,v)-ev(t-step,v))/(2*step),
                       (ev(t,v+step)-ev(t,v-step))/(2*step)])
        errors['V_gradient'].append(float(np.max(np.abs(grad-fd))))
        psi = v+np.pi
        old = np.array([[-np.sin(t),0,np.cos(t)],[-np.sin(t),0,np.cos(t)],
                        [np.sin(psi),0,-np.cos(psi)]])
        current=np.array([[np.sin(t),0,np.cos(t)],[np.sin(t),0,np.cos(t)],
                          [np.sin(v),0,np.cos(v)]])
        errors['historical_Gamma_to_V_geometry'].append(float(np.max(np.abs(old@np.diag([-1,-1,1])-current))))

    h=.2
    def eu(x):
        # Exact fixed-length Cartesian path about UUD; expansion coordinate is sqrt(S)*x.
        spins=np.column_stack((np.sqrt(spin)*x, np.zeros(3),
                               np.array([1,1,-1])*np.sqrt(spin**2-spin*x*x)))
        return sum(j*np.dot(spins[a,:2],spins[b,:2])+jz*spins[a,2]*spins[b,2]
                   for a,b in [(0,1),(1,2),(2,0)])-h*spins[:,2].sum()/3
    expected=np.array([[h/3,j*spin,j*spin],[j*spin,h/3,j*spin],
                       [j*spin,j*spin,2*jz*spin-h/3]])
    step=1e-4
    fd=np.zeros((3,3)); zero=np.zeros(3)
    for a in range(3):
        xa=np.eye(3)[a]*step
        fd[a,a]=(eu(xa)+eu(-xa)-2*eu(zero))/step**2
        for b in range(a):
            xb=np.eye(3)[b]*step
            fd[a,b]=fd[b,a]=(eu(xa+xb)-eu(xa-xb)-eu(-xa+xb)+eu(-xa-xb))/(4*step**2)
    errors['UUD_Cartesian_Hessian'].append(float(np.max(np.abs(fd-expected))))
    maxima={name:max(values) for name,values in errors.items()}
    assert maxima['V_gradient'] < 1e-8 and maxima['UUD_Cartesian_Hessian'] < 1e-7
    assert all(maxima[name]<1e-14 for name in ['three_sublattice_energy','stripe_energy','historical_Gamma_to_V_geometry'])
    assert all(row['UUD_inside_min']>0 and max(row['UUD_outside_min'])<0 for row in rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result={'scope':'Local algebra and field conversion; not global phase stability',
            'parameters':{'S':spin,'Jz_meV':jz,'gz':gz,'muB_meV_per_T':mu_b},
            'critical_fields':rows,'maximum_absolute_residuals':maxima,
            'random_seed':916,'random_samples':20,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT/'phase-source-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
