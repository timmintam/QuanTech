import numpy as np

from qiskit_nature.settings import settings
settings.dict_aux_operators = True

from qiskit_nature.properties import Property, GroupedProperty
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.opflow import TwoQubitReduction
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    #ElectronicDipoleMoment,
    #ParticleNumber,
    #AngularMomentum,
    #Magnetization,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    #ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    #IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis

from qiskit.algorithms import VQE
from qiskit_nature.circuit.library import UCCSD, HartreeFock
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




def get_qubit_operators(h2_1e_MO: np.array, h2_2e_MO: np.array, nuc_rep_energy: float, num_particles: tuple):
    
    electronic_energy = ElectronicEnergy.from_raw_integrals(
            ElectronicBasis.MO,h2_1e_MO, h2_2e_MO,
        )
    electronic_energy.nuclear_repulsion_energy = nuc_rep_energy

    hamiltonian = electronic_energy.second_q_ops()["ElectronicEnergy"]
    print("Electronic part of the Hamiltonian :\n", hamiltonian)
    

    # comment : how to freeze the core orbitals ???
    
    
    mapper = ParityMapper()  # Set Mapper
    # Do two qubit reduction
    converter = QubitConverter(mapper,two_qubit_reduction=True)
    qubit_op = converter.convert_only(hamiltonian, num_particles)
    print("q_op :\n", qubit_op)
   

    return qubit_op, converter





def calc_ground_state(op,num_part,num_orb,converter):

    backend = BasicAer.get_backend("statevector_simulator")
 
    #result = exact_solver(problem,converter)

    optimizer = SLSQP(maxiter=5)
    
    init_state = HartreeFock(num_orb, num_part, converter)
    print(init_state) # WHY DOES IT INITIALIZE A STATE WITH 4 QUBITS ??? INSTEAD OF 2...
     
    var_form = UCCSD(converter,
                        num_part,
                        num_orb,
                        initial_state=init_state)

    vqe = VQE(var_form, optimizer, quantum_instance=backend) 
    vqe_result = vqe.compute_minimum_eigenvalue(op)
    min_eng = vqe_result.eigenvalue
    #vqe_ground = vqe_result.eigenstate perhaps more accurate? Downside: don't get circuit 
    final_params = vqe_result.optimal_parameters 

    vqe_ground = vqe.ansatz.bind_parameters(final_params)  
    
    return vqe_ground, min_eng