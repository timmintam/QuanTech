import numpy as np
from molecular_integrals import get_molecular_integrals
from pyscf_qiskit import QiskitNaturePySCFSolver

coordinates=np.array([1.0])
dR=np.array([0.1])

solver = QiskitNaturePySCFSolver()

# Option 1:
#h1_MO, h2_MO, num_orb, num_particles, nuc_rep_energy = get_molecular_integrals(coordinates)
#solver.kernel(h2_1e_MO, h2_2e_MO, num_orb, num_particles, nuc_rep_energy)

# Option 2:
H_0 = solver.kernel(*get_molecular_integrals(coordinates))

# Option 3:
# TODO define a Results structure

vqe_ground, min_eng = solver.calc_ground_state()
print(f'Ground state preparation circuit : \n{vqe_ground}')
print(f'Ground energy : {min_eng}')


H_plus  = solver.kernel(*get_molecular_integrals(coordinates+dR))
H_minus = solver.kernel(*get_molecular_integrals(coordinates-dR))
print(f'\n\nCompare \n\n{(H_plus-H_minus)}\n\n versus \n\n{(H_plus-H_minus).reduce()}')
obs=(H_plus-H_minus).reduce()
print(obs.primitive.to_list())
