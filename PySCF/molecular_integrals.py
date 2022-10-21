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

    # computing the 1e and 2e AO (atomic orbitals) integrals 
    h1 = mol_h2.intor_symmetric("int1e_nuc") + mol_h2.intor_symmetric("int1e_kin") 
    h2 = mol_h2.intor("int2e")
    # comment : add aom (axis of symmetry) ???

    # transformations from AO to MO (molecular orbitals)
    h1_MO = np.einsum('pi,pq,qj->ij', mf.mo_coeff, h1, mf.mo_coeff)
    h2_MO = ao2mo.get_mo_eri(h2, mf.mo_coeff)

    #h2_MO = ao2mo.kernel(mol_h2, mf.mo_coeff, aosym="1") #alternative
    
    nuclear_repulsion_energy=mol_h2.energy_nuc()
    
    ##########################################
    num_particles = mol_h2.nelec # correct ???
    ##########################################
    num_orb = h1.shape[0]        # correct ???
    ##########################################
    print(f'Number of particles : {num_particles}')
    print(f'Number of spin orbitals : {num_orb}')
    
    return h1_MO, h2_MO, num_orb, num_particles, nuclear_repulsion_energy