# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pyscf import ao2mo, gto, scf


#from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolvers
#from qiskit_nature.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)

from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ParticleNumber,
    #ElectronicDensity
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem

"""
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
)

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import (
    ElectronicDensity,
    ParticleNumber,
)
"""

class QiskitNaturePySCFSolver:
    def __init__(self, solver: GroundStateEigensolver):
        #self.density: ElectronicDensity = None
        self.solver = solver

    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        hamiltonian = ElectronicEnergy.from_raw_integrals(
            h1, ao2mo.restore(1, h2, norb)
        )
        hamiltonian.nuclear_repulsion_energy = ecore
        problem = ElectronicStructureProblem(hamiltonian)
        problem.num_spatial_orbitals = norb
        problem.num_particles = nelec
        #if self.density is None:
        #    self.density = ElectronicDensity.from_orbital_occupation(
        #        problem.orbital_occupations,
        #        problem.orbital_occupations_b,
        #    )
        problem.properties.particle_number = ParticleNumber(norb)
        problem.properties.electronic_density = self.density

        self.result = self.solver.solve(problem)
        #self.density = self.result.electronic_density

        e_tot = self.result.total_energies[0]
        return e_tot, self
    
    def get_qubit_ops():
        return 0
        

    """
    def make_rdm1(self, fake_ci, norb, nelec):
        return fake_ci.density.trace_spin()["+-"]

    def make_rdm12(self, fake_ci, norb, nelec):
        traced = fake_ci.density.trace_spin()
        return (traced["+-"], _phys_to_chem(traced["++--"]))
    """

"""
if __name__ == "__main__":
    mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="631g*", spin=0, verbose=4)
    mf = scf.RHF(mol).run()
    norb = 2
    nelec = 2
    mc = mcscf.CASCI(mf, norb, nelec)

    solver = GroundStateEigensolver(
        QubitConverter(JordanWignerMapper()),
        NumPyMinimumEigensolverFactory(),
    )

    mc.fcisolver = QiskitNaturePySCFFCISolver(solver)
    mc.run()
"""