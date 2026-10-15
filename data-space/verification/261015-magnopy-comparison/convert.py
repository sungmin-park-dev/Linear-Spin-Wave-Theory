"""Convert a spin-toolkit (SpinModel, SpinState, field) to a magnopy LSWT object."""
import numpy as np
import magnopy

def to_magnopy(model, state, field=(0, 0, 0)):
    M = np.array(state.supercell, dtype=int)
    Minv = np.linalg.inv(M)
    cells = list(state.cells)
    keys = [(s.id, c) for c in cells for s in model.sites]
    index = {k: i for i, k in enumerate(keys)}
    spin = {s.id: s.spin for s in model.sites}
    pos = {s.id: np.array(s.position) for s in model.sites}
    A2 = state.magnetic_lattice(model)
    cell3 = np.zeros((3, 3)); cell3[:2, :2] = A2; cell3[2, 2] = 20.0
    # fractional positions w.r.t. magnetic cell
    atoms = {"names": [], "positions": [], "spins": [], "g_factors": []}
    for (sid, c) in keys:
        cart = (np.array(c) + pos[sid]) @ model.lattice
        frac = np.linalg.solve(A2.T, cart)
        atoms["names"].append(f"{sid}_{c[0]}_{c[1]}")
        atoms["positions"].append([frac[0], frac[1], 0.0])
        atoms["spins"].append(spin[sid])
        atoms["g_factors"].append(2.0)
    conv = magnopy.Convention(multiple_counting=False, spin_normalized=False,
                              c1=1.0, c21=1.0, c22=1.0, name="stk")
    H = magnopy.SpinHamiltonian(cell=cell3, atoms=atoms, convention=conv)

    def split(sid, g):
        r = state.reduce_cell(g)
        t = np.rint((np.array(g) - np.array(r)) @ Minv).astype(int)
        return index[(sid, r)], (int(t[0]), int(t[1]), 0)

    b = np.array(field, float)
    for term in model.terms:
        for c in cells:
            if term.kind == "bilinear":
                (s1, n1), (s2, n2) = term.participants
                a1, t1 = split(s1, np.add(c, n1)); a2, t2 = split(s2, np.add(c, n2))
                H.add(parameter=np.array(term.coefficient), alphas=[a1, a2], nus=[t1, t2],
                      when_present="sum")
            elif term.kind == "zeeman":
                (s1, n1), = term.participants
                a1, t1 = split(s1, np.add(c, n1))
                H.add(parameter=-(np.array(term.coefficient).T @ b), alphas=[a1], nus=[t1],
                      when_present="sum")
            elif term.kind == "onsite":
                (s1, n1), = term.participants
                a1, t1 = split(s1, np.add(c, n1))
                H.add(parameter=np.array(term.coefficient), alphas=[a1, a1], nus=[t1, t1],
                      when_present="sum")
            else:
                raise ValueError(term.kind)
    dirs = [state.direction(sid, c) for (sid, c) in keys]
    return magnopy.LSWT(spinham=H, spin_directions=dirs), keys, A2
