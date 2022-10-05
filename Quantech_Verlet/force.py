from hamiltonian import get_qubit_op
from energy import calc_energy

eV = 1.602e-19
angst = 1e-10
dR = 0.02*angst

def calc_forces(coordinates):
    
    #get qubit operators coresponding to H+ (hamiltonian for atomic distances R + dR)
    (qubit_op_plus_0, _, _, _, _) = get_qubit_op([coordinates[0] + dR,coordinates[1]])
    
    #get qubit operators coresponding to H- (hamiltonian for atomic distances R - dR)
    (qubit_op_minus_0, num_particles, num_spin_orbitals, problem, converter) = get_qubit_op([coordinates[0] - dR,coordinates[1]])
    
    #get minimum eigenvalue of (H_+ - H_-)/(2dR) = force
    f0 = calc_energy(qubit_op_plus_0-qubit_op_minus_0,num_particles,num_spin_orbitals,problem,converter)/(2*dR)*eV


    #repeat same for other atom
    (qubit_op_plus_1, _, _, _, _) = get_qubit_op([coordinates[0],coordinates[1]+dR])
    
    (qubit_op_minus_1, num_particles, num_spin_orbitals, problem, converter) = get_qubit_op([coordinates[0],coordinates[1]-dR])

    f1 = calc_energy(qubit_op_plus_1-qubit_op_minus_1,num_particles,num_spin_orbitals,problem,converter)/(2*dR)*eV


    return f0,f1