from hamiltonian import get_qubit_op
from energy import calc_energy, calc_ground_state
import numpy as np
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, CircuitStateFn
from qiskit import Aer



eV = 1.602e-19
angst = 1e-10
dR = 0.002*angst

Ha = 4.36e-18
angst = 1.0e-10
dR = 0.02 #removed the angst, since coordinates are given in angst

dx = np.array([dR,0,0])
dy = np.array([0,dR,0])
dz = np.array([0,0,dR])

def calc_forces(coordinates):

    #get qubit operator corresponding to H(R)
    (qubit_op, num_part, num_orb, problem, converter) = get_qubit_op([coordinates[0],coordinates[1]])

    #get qubit operators coresponding to H+ (hamiltonian for atomic distances R + idR, R + jdR, R + kdR )
    (qubit_op_plus_0_x, _, _, problem_plus_0_x, converter_plus_0_x) = get_qubit_op([coordinates[0] + dx,coordinates[1]])
    (qubit_op_plus_0_y, _, _, problem_plus_0_y, converter_plus_0_y) = get_qubit_op([coordinates[0] + dy,coordinates[1]])
    (qubit_op_plus_0_z, _, _, problem_plus_0_z, converter_plus_0_z) = get_qubit_op([coordinates[0] + dz,coordinates[1]])
    
    #get qubit operators coresponding to H- (hamiltonian for atomic distances R - idR, R - jdR, R - kdR )
    (qubit_op_minus_0_x, num_part_x, num_orb_x, problem_minus_0_x, converter_minus_0_x) = get_qubit_op([coordinates[0] - dx,coordinates[1]])
    (qubit_op_minus_0_y, num_part_y, num_orb_y, problem_minus_0_y, converter_minus_0_y) = get_qubit_op([coordinates[0] - dy,coordinates[1]])
    (qubit_op_minus_0_z, num_part_z, num_orb_z, problem_minus_0_z, converter_minus_0_z) = get_qubit_op([coordinates[0] - dz,coordinates[1]])
 
    #get the ground state of H(R)
    psi_0,_ = calc_ground_state(qubit_op,num_part, num_orb, problem, converter)
    
    #define desired observable (H_+ - H_-)/(2dR) = force  
    Obs0_x = (qubit_op_plus_0_x-qubit_op_minus_0_x)
    Obs0_y = (qubit_op_plus_0_y-qubit_op_minus_0_y)
    Obs0_z = (qubit_op_plus_0_z-qubit_op_minus_0_z)
       
    #get the expectation value <psi_0|O|psi_0>¨
    
    backend = Aer.get_backend('qasm_simulator') 
    q_instance = QuantumInstance(backend, shots=8024)

    
    psi_0 = CircuitStateFn(psi_0)
    '''
    # Option 1: Calucate the force according to Equation 7
    measurable_expression = StateFn(Obs0, is_measurement=True).compose(psi_0) 
    expectation = PauliExpectation().convert(measurable_expression)  
    sampler = CircuitSampler(q_instance).convert(expectation) 
    f0 = sampler.eval().real
    '''
    
    #Option 2: Calculate the force according to Equation 6
    
    f0_plus_x = calc_energy(qubit_op_plus_0_x,num_part_x,num_orb_x,problem_plus_0_x,converter_plus_0_x)
    f0_minus_x = calc_energy(qubit_op_minus_0_x,num_part_x,num_orb_x,problem_minus_0_x,converter_minus_0_x)

    f0_plus_y = calc_energy(qubit_op_plus_0_y,num_part_y,num_orb_y,problem_plus_0_y,converter_plus_0_y)
    f0_minus_y = calc_energy(qubit_op_minus_0_y,num_part_y,num_orb_y,problem_minus_0_y,converter_minus_0_y)

    f0_plus_z = calc_energy(qubit_op_plus_0_z,num_part_z,num_orb_z,problem_plus_0_z,converter_plus_0_z)
    f0_minus_z = calc_energy(qubit_op_minus_0_z,num_part_z,num_orb_z,problem_minus_0_z,converter_minus_0_z)

    f0_x = f0_plus_x - f0_minus_x
    f0_y = f0_plus_y - f0_minus_y
    f0_z = f0_plus_z - f0_minus_z
    
    
    ### repeat same for other atom ###
    (qubit_op_plus_1_x, _, _, _, _) = get_qubit_op([coordinates[0],coordinates[1]+dx])
    (qubit_op_plus_1_y, _, _, _, _) = get_qubit_op([coordinates[0],coordinates[1]+dy])
    (qubit_op_plus_1_z, _, _, _, _) = get_qubit_op([coordinates[0],coordinates[1]+dz])
    
    (qubit_op_minus_1_x, num_particles_x, num_spin_orbitals_x, problem_x, converter_x) = get_qubit_op([coordinates[0],coordinates[1]-dx])
    (qubit_op_minus_1_y, num_particles_y, num_spin_orbitals_y, problem_y, converter_y) = get_qubit_op([coordinates[0],coordinates[1]-dy])
    (qubit_op_minus_1_z, num_particles_z, num_spin_orbitals_z, problem_z, converter_z) = get_qubit_op([coordinates[0],coordinates[1]-dz])

    #define desired observable (H_+ - H_-)/(2dR) = force  
    Obs1_x = (qubit_op_plus_1_x-qubit_op_minus_1_x)
    Obs1_y = (qubit_op_plus_1_y-qubit_op_minus_1_y)
    Obs1_z = (qubit_op_plus_1_z-qubit_op_minus_1_z)
    
    backend = Aer.get_backend('qasm_simulator') 
    q_instance = QuantumInstance(backend, shots=1)
    
    measurable_expression_x = StateFn(Obs1_x, is_measurement=True).compose(psi_0)
    measurable_expression_y = StateFn(Obs1_y, is_measurement=True).compose(psi_0) 
    measurable_expression_z = StateFn(Obs1_z, is_measurement=True).compose(psi_0)
    
    expectation_x = PauliExpectation().convert(measurable_expression_x)
    expectation_y = PauliExpectation().convert(measurable_expression_y)  
    expectation_z = PauliExpectation().convert(measurable_expression_z)
    
    sampler_x = CircuitSampler(q_instance).convert(expectation_x)
    sampler_y = CircuitSampler(q_instance).convert(expectation_y) 
    sampler_z = CircuitSampler(q_instance).convert(expectation_z) 
    
    f1_x = sampler_x.eval().real
    f1_y = sampler_y.eval().real
    f1_z = sampler_z.eval().real

    f0 = np.array([f0_x,f0_y,f0_z])*Ha*(1/(2*dR*angst))
    f1 = np.array([f1_x,f1_y,f1_z])*Ha*(1/(2*dR*angst))
    
    return f0,f1
