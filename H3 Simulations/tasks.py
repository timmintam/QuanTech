
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.quantum_info import Pauli

def task1(string, q_instance, psi_0):

    Obs = PauliOp(Pauli(string))
    measurable_expression = StateFn(Obs, is_measurement=True).compose(psi_0)
    expectation = PauliExpectation().convert(measurable_expression)  
    sampler = CircuitSampler(q_instance).convert(expectation) 
    E = sampler.eval().real
    
    
        
    return E

