import numpy as np

from pyscf import ao2mo

from typing import Tuple

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.properties import ParticleNumber
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.problems import ElectronicBasis
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo

from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit import Aer, BasicAer
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.operators import Operator

class QiskitNaturePySCFSolver:
    """
    A class to get:
        1) qubit operators from raw 1e and 2e integrals
        2) ground state using VQE
    """
    
    def __init__(self,
                 #molecule: MoleculeInfo
    ):
        
        #self.molecule = molecule
        self.qubit_op  = None
        self.converter = None
        self.norb = None
        self.nelec = None



    
    def kernel(self, 
               h1_MO: np.array,
               h2_MO: np.array, 
               norb: int,
               nelec: Tuple[int, int],
               #coords: Sequence[tuple[float, float, float]],
               ecore: float = 0.0,
               **kwargs
    ) -> Operator:
        
        self.norb = norb
        self.nelec = nelec

        # Define an ElectronicEnergy instance containing the 1e and 2e integrals
        electronic_energy = ElectronicEnergy.from_raw_integrals(
                h1_MO, ao2mo.restore(1, h2_MO, self.norb)
            )
        electronic_energy.nuclear_repulsion_energy = ecore

        # Define an ElectronicStructureProblem
        problem = ElectronicStructureProblem(electronic_energy)

        second_q_ops = problem.second_q_ops()     # get second quantized operators
        problem.num_spatial_orbitals = self.norb  # define number of orbitals
        problem.num_particles = self.nelec        # define number of particles 
        
        problem.basis = ElectronicBasis.MO # 1e and 2e integrals are expected to be given in the Molecular Orbitals basis
        # TODO raise error if h1 and h2 not in MO basis ? how to check ? add argument to pass the basis ?
        
        
        #problem.molecule=MoleculeInfo(symbols=('H','H'), coords=((0.0, 0.0, 0.0),(1.0, 0.0, 0.0)) )
        #print(problem.molecule)
        #FC_transformer=FreezeCoreTransformer(freeze_core=True)
        #problem = FC_transformer.transform(problem)
        # TODO we need to give info about the molecule to use FreezeCoreTransformer 
        # question : how to do it in a smart way ? at the initialization ??
        
        hamiltonian = second_q_ops[0]  # Set electronic part of Hamiltonian
        hamiltonian = (hamiltonian + FermionicOp({"": ecore}, num_spin_orbitals=self.norb)) # add Nuclear Repulsion Energy
        #print("Hamiltonian :\n", hamiltonian) # print for checking purposes
        
        mapper = ParityMapper()  # Set Mapper
        
        # Do two qubit reduction
        converter = QubitConverter(mapper,two_qubit_reduction=True)
        qubit_op = converter.convert(hamiltonian, self.nelec)
        #print("q_op :\n", qubit_op) # print for checking purposes
        
        self.qubit_op  = qubit_op
        self.converter = converter
    
        return qubit_op
        
    

    def calc_ground_state(self) -> Tuple[QuantumCircuit, complex] :
        
        # question : vqe_ground is of type QuantumCircuit right ?
        # TODO raise error if kernel has not been called previously
        
        backend = Aer.get_backend('qasm_simulator') 
        quantum_instance = QuantumInstance(backend, shots=128)
        optimizer = SPSA(maxiter=100)
        
        init_state = HartreeFock(num_spatial_orbitals=self.norb, 
                                 num_particles=self.nelec, 
                                 qubit_converter=self.converter
                                )
        
        var_form = UCCSD(qubit_converter=self.converter,
                         num_particles=self.nelec,
                         num_spatial_orbitals=self.norb, 
                         initial_state=init_state
                         )

        vqe = VQE(var_form, optimizer, quantum_instance=quantum_instance) 
        vqe_result = vqe.compute_minimum_eigenvalue(self.qubit_op)
        min_eng = vqe_result.eigenvalue
        final_params = vqe_result.optimal_parameters 

        vqe_ground = vqe.ansatz.bind_parameters(final_params)  
        
        return vqe_ground, min_eng
