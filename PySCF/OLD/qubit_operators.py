import numpy as np

from pyscf import ao2mo

from qiskit_nature.settings import settings
settings.dict_aux_operators = True


from typing import Tuple

from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.properties import (
    ElectronicDensity,
    ParticleNumber,
)
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.problems import ElectronicBasis
#from qiskit_nature.second_q.formats import MoleculeInfo
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

###########################################################################

from qiskit_nature.properties import Property, GroupedProperty
from qiskit.opflow import TwoQubitReduction

#from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer

from qiskit.algorithms import VQE
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit import IBMQ, BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
#from qiskit.providers.aer.noise import NoiseModel
#from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, CircuitStateFn
from qiskit.quantum_info.operators import Operator




def get_qubit_operators(h2_1e_MO: np.array, h2_2e_MO: np.array, nuc_rep_energy: float, num_particles: Tuple[int, int], norb: int) -> Tuple[Operator, QubitConverter] :
    
    electronic_energy = ElectronicEnergy.from_raw_integrals(
            h2_1e_MO, ao2mo.restore(1, h2_2e_MO, norb)
        )
    electronic_energy.nuclear_repulsion_energy = nuc_rep_energy

    problem = ElectronicStructureProblem(electronic_energy)
    
    second_q_ops = problem.second_q_ops()
    problem.num_spatial_orbitals = norb
    problem.num_particles = num_particles
    
    problem.basis=ElectronicBasis.MO
    problem.molecule=MoleculeInfo(symbols=('H','H'), coords=((0.0, 0.0, 0.0),(1.0, 0.0, 0.0)) )
    print(problem.molecule)
    FC_transformer=FreezeCoreTransformer(freeze_core=True)
    problem = FC_transformer.transform(problem)
    
    
    hamiltonian=second_q_ops[0]  # Set Hamiltonian
    print("Electronic part of the Hamiltonian :\n", hamiltonian)
    
    
    mapper = ParityMapper()  # Set Mapper
    # Do two qubit reduction
    converter = QubitConverter(mapper,two_qubit_reduction=True)
    qubit_op = converter.convert(hamiltonian, num_particles)
    print("q_op :\n", qubit_op)
   

    return (qubit_op, converter)





def calc_ground_state(op,num_part,num_orb,converter):

    backend = BasicAer.get_backend("statevector_simulator")
 
    #result = exact_solver(problem,converter)

    optimizer = SLSQP(maxiter=5)
    
    print(num_orb, num_part)
    init_state = HartreeFock(num_spatial_orbitals=num_orb, num_particles=num_part, qubit_converter=converter)
    print(init_state) 
     
    var_form = UCCSD(qubit_converter=converter,
                        num_particles=num_part,
                        num_spatial_orbitals=num_orb,
                        initial_state=init_state)

    vqe = VQE(var_form, optimizer, quantum_instance=backend) 
    vqe_result = vqe.compute_minimum_eigenvalue(op)
    min_eng = vqe_result.eigenvalue
    #vqe_ground = vqe_result.eigenstate perhaps more accurate? Downside: don't get circuit 
    final_params = vqe_result.optimal_parameters 

    vqe_ground = vqe.ansatz.bind_parameters(final_params)  
    
    return vqe_ground, min_eng