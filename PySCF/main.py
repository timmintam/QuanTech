import numpy as np
from molecular_integrals import get_molecular_integrals
from qubit_operators import get_qubit_operators, calc_ground_state
from pyscf_qiskit import QiskitNaturePySCFSolver

"""
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit_nature.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
"""

coordinates=np.array([1.0])
h2_1e_MO, h2_2e_MO, nuc_rep_energy, num_particles, num_orb = get_molecular_integrals(coordinates)
print(num_particles)
qubit_op, converter = get_qubit_operators(h2_1e_MO, h2_2e_MO, nuc_rep_energy, num_particles)
print("Qubit operators :\n",qubit_op)
vqe_ground, min_eng = calc_ground_state(qubit_op,num_particles,num_orb,converter)
print(min_eng)



#mapper = ParityMapper()  # Set Mapper
# Do two qubit reduction
#converter = QubitConverter(mapper,two_qubit_reduction=True)
#numpy_solver = NumPyMinimumEigensolverFactory()
#GSE = GroundStateEigensolver(converter, numpy_solver)
#solver= QiskitNaturePySCFSolver(GSE)