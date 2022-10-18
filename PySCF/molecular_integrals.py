from pyscf import gto, scf, ao2mo
import numpy as np

def get_molecular_integrals(coordinates: np.array):

    # define the molecule
    mol_h2 = gto.M(
        atom = [['H',(0.0, 0.0, 0.0)], 
                ['H',(coordinates[0], 0.0, 0.0)]],
        basis = 'sto3g',
        charge = 0,
        spin = 0,
        symmetry = False,
    )

    mf = scf.RHF(mol_h2).run()
    
    # mf.frozen = 1
    # comment : frozen doesn't work... how to freeze the core orbitals ???

    # computing the 1e and 2e AO (atomic orbitals) integrals 
    h2_1e= mol_h2.intor_symmetric("int1e_nuc") + mol_h2.intor_symmetric("int1e_kin") # + mol_h2.intor_symmetric("int1e_ovlp") 
    h2_2e=mol_h2.intor("int2e")
    # comment : add aom (axis of symmetry) ???

    # transformations from AO to MO (molecular orbitals)
    h2_1e_MO=np.einsum('pi,pq,qj->ij', mf.mo_coeff, h2_1e, mf.mo_coeff)
    h2_2e_MO=ao2mo.get_mo_eri(h2_2e, mf.mo_coeff)

    # mo_ints = ao2mo.kernel(mol_h2, mf.mo_coeff, aosym="1")
    # comment : why this method does not give the correct shape ?
    
    nuclear_repulsion_energy=mol_h2.energy_nuc()
    
    ##########################################
    num_particles = mol_h2.nelec # correct ???
    ##########################################
    num_orb = (mf.mo_coeff).size # correct ???
    ##########################################
    print(f'Number of particles : {num_particles}')
    print(f'Number of spin orbitals : {num_orb}')
    
    return h2_1e_MO, h2_2e_MO, nuclear_repulsion_energy, num_particles, num_orb

