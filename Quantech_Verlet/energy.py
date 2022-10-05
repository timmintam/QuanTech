from qiskit.algorithms import VQE
from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)
import matplotlib.pyplot as plt
import numpy as np
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit import IBMQ, BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel

#not needed
#def exact_solver(problem, converter):
    #solver = NumPyMinimumEigensolverFactory()
    #calc = GroundStateEigensolver(converter, solver)
    #result = calc.solve(problem)
    #return result

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

    #not needed
    #distances = np.arange(0.5, 4.0, 0.2)
    #not needed
    #exact_energies = []
    #vqe_energies = []
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
