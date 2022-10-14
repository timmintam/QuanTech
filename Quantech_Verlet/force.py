from hamiltonian import get_qubit_op
from energy import calc_ground_state
import numpy as np
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, CircuitStateFn
from qiskit import Aer



eV = 1.602e-19
angst = 1e-10
dR = 0.002*angst

seed = 50

Ha = 4.36e-18
angst = 1.0e-10

#number of shots to calculate covariance matrix
m = 10

dR = 0.02 #removed the angst, since coordinates are given in angst

dx = np.array([dR,0,0])
dy = np.array([0,dR,0])
dz = np.array([0,0,dR])


def get_psi_0(coordinates):

    #get qubit operator corresponding to H(R)
    (qubit_op, num_part, num_orb, problem, converter) = get_qubit_op(coordinates)
    #get the ground state of H(R)
    psi_0,_ = calc_ground_state(qubit_op,num_part, num_orb, problem, converter)
    psi_0 = CircuitStateFn(psi_0)

    return psi_0

#function to calculate covariance matrix

def calc_cov(coordinates):
    #calculate psi_0 at distances R for hamiltonian H(R)
    psi_0 = get_psi_0(coordinates)

    #array to hold force components    
    force_components = []
    for i in range(0,m):
        #calculate forces for psi_0    
        forces_i = calc_forces(coordinates,psi_0)
        #temp array
        new_forces = []
        #transform [[f_x0,f_y0,f_z0],[f_x1,f_y1,f_z1]] to [f_x0,f_y0,f_z0,f_x1,f_y1,f_z1] and append it to force_components
        for j in range(len(forces_i)):
            new_forces = np.concatenate((new_forces,forces_i[j]),axis=None)
        force_components.append(new_forces)            

    #take the transpose of force_components
    force_components = np.array(force_components).T
    #calculate the covariance matrix
    covariance_matrix = np.cov(force_components)                



    return covariance_matrix

def calc_forces(coordinates, psi = None):

    backend = Aer.get_backend('qasm_simulator')

    psi_0 = psi

    if (psi_0 == None):
        psi_0 = get_psi_0(coordinates)

    forces = []
    #loop over particles
    
    for i in range(len(coordinates)):

    
        #get qubit operators coresponding to H+ (hamiltonian for atomic distances R + idR, R + jdR, R + kdR )
        (qubit_op_plus_x, _, _, problem_plus_x, converter_plus_x) = get_qubit_op(coordinates[0:i] + [(coordinates[i] + dx)] + coordinates[i+1:])
        (qubit_op_plus_y, _, _, problem_plus_y, converter_plus_y) = get_qubit_op(coordinates[0:i] + [(coordinates[i] + dy)] + coordinates[i+1:])
        (qubit_op_plus_z, _, _, problem_plus_z, converter_plus_z) = get_qubit_op(coordinates[0:i] + [(coordinates[i] + dz)] + coordinates[i+1:])
        
        #get qubit operators coresponding to H- (hamiltonian for atomic distances R - idR, R - jdR, R - kdR )
        (qubit_op_minus_x, num_part_x, num_orb_x, problem_minus_x, converter_minus_x) = get_qubit_op(coordinates[0:i] + [(coordinates[i] - dx)] + coordinates[i+1:])
        (qubit_op_minus_y, num_part_y, num_orb_y, problem_minus_y, converter_minus_y) = get_qubit_op(coordinates[0:i] + [(coordinates[i] - dy)] + coordinates[i+1:])
        (qubit_op_minus_z, num_part_z, num_orb_z, problem_minus_z, converter_minus_z) = get_qubit_op(coordinates[0:i] + [(coordinates[i] - dz)] + coordinates[i+1:])
        #get observables
        Obs_x = (qubit_op_plus_x-qubit_op_minus_x)
        Obs_y = (qubit_op_plus_y-qubit_op_minus_y)
        Obs_z = (qubit_op_plus_z-qubit_op_minus_z)

    
        #get nuclear repulsion energies
        rep_eng_plus_x = problem_plus_x.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
        rep_eng_plus_y = problem_plus_y.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
        rep_eng_plus_z = problem_plus_z.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy

        rep_eng_minus_x = problem_minus_x.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
        rep_eng_minus_y = problem_minus_y.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
        rep_eng_minus_z = problem_minus_z.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
         
        q_instance = QuantumInstance(backend, shots=1024, seed_transpiler=seed, seed_simulator=seed)

        measurable_expression_x = StateFn(Obs_x, is_measurement=True).compose(psi_0)
        measurable_expression_y = StateFn(Obs_y, is_measurement=True).compose(psi_0)
        measurable_expression_z = StateFn(Obs_z, is_measurement=True).compose(psi_0)

        expectation_x = PauliExpectation().convert(measurable_expression_x)
        expectation_y = PauliExpectation().convert(measurable_expression_y)
        expectation_z = PauliExpectation().convert(measurable_expression_z)

        sampler_x = CircuitSampler(q_instance).convert(expectation_x)
        sampler_y = CircuitSampler(q_instance).convert(expectation_y) 
        sampler_z = CircuitSampler(q_instance).convert(expectation_z)

        f_x = sampler_x.eval().real + rep_eng_plus_x - rep_eng_minus_x
        f_y = sampler_y.eval().real + rep_eng_plus_y - rep_eng_minus_y
        f_z = sampler_z.eval().real + rep_eng_plus_z - rep_eng_minus_z

        forces.append(np.array([f_x,f_y,f_z])*Ha*(1/(2*dR*angst)))
    
    return forces
