from qiskit.algorithms import VQE
from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)
import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit import IBMQ, BasicAer, Aer
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.operators import Operator



def exact_solver(problem, converter):
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    return result

def calc_energy(op,num_part,num_orb,problem,converter):
    
    backend = BasicAer.get_backend("statevector_simulator")

    #no clue why this is needed. Without it the initial state has different # of qubits than the number of qubits in qubit operator
    # and we get an error. 
    result = exact_solver(problem,converter)

    optimizer = SLSQP(maxiter=5)

        #result = exact_solver(problem,converter)
        #exact_energies.append(result.total_energies[0].real)
    
    init_state = HartreeFock(num_orb, num_part, converter)
   
    var_form = UCCSD(converter,
                        num_part,
                        num_orb,
                        initial_state=init_state)
    vqe = VQE(var_form, optimizer, quantum_instance=backend)
    
    vqe_calc = vqe.compute_minimum_eigenvalue(op)
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    return vqe_result 

def calc_ground_state(op,num_part,num_orb,problem,converter):

    backend = BasicAer.get_backend("statevector_simulator")
 
    result = exact_solver(problem,converter)

    optimizer = SLSQP(maxiter=5)

    init_state = HartreeFock(num_orb, num_part, converter)
     
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
