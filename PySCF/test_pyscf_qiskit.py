from pyscf import gto, scf, ao2mo
import numpy as np

coordinates=np.array([0.0,1.0])

# define the molecule
mol_h2 = gto.M(
    atom = [['H',(coordinates[0], 0.0, 0.0)], 
            ['H',(coordinates[1], 0.0, 0.0)]],
    basis = 'sto3g',
    charge = 0,
    spin = 0,
    symmetry = False,
)



mf = scf.RHF(mol_h2)
mf.frozen = 1
mf.run()

# computing the 1e and 2e AO (atomic orbitals) integrals 

h2_1e= mol_h2.intor_symmetric("int1e_nuc") + mol_h2.intor_symmetric("int1e_kin") # + mol_h2.intor_symmetric("int1e_ovlp") 
#comment : add aom (axis of symmetry) ???
print(h2_1e)

h2_2e=mol_h2.intor("int2e")
#comment : add aom (axis of symmetry) ???
print(h2_2e)

# transformations from AO to MO (molecular orbitals)
h2_1e_MO=np.einsum('pi,pq,qj->ij', mf.mo_coeff, h2_1e, mf.mo_coeff)
mo_ints = ao2mo.kernel(mol_h2, mf.mo_coeff, aosym="1")
print(mo_ints)
h2_2e_MO=ao2mo.get_mo_eri(h2_2e, mf.mo_coeff)


from qiskit_nature.settings import settings

settings.dict_aux_operators = True

from qiskit_nature.properties import Property, GroupedProperty

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit.opflow import TwoQubitReduction




from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ParticleNumber,
    AngularMomentum,
    Magnetization,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis


one_body_ints = OneBodyElectronicIntegrals(
    ElectronicBasis.MO,
    (
        h2_1e_MO,h2_1e_MO,
    ),
)
print(one_body_ints)

two_body_ints = TwoBodyElectronicIntegrals(
    ElectronicBasis.MO,
    (
        h2_2e_MO, h2_2e_MO, h2_2e_MO, h2_2e_MO, 
    ),
)
print(two_body_ints)

electronic_energy = ElectronicEnergy(
    [one_body_ints, two_body_ints],
    nuclear_repulsion_energy=mol_h2.energy_nuc(),
    
)
print(electronic_energy.nuclear_repulsion_energy)

hamiltonian = electronic_energy.second_q_ops()["ElectronicEnergy"]
print(hamiltonian)


##########################################
num_particles = mol_h2.nelec # correct ???
##########################################
num_orb = (mf.mo_coeff).size # correct ???
##########################################
print(num_orb)

# comment : how to freeze the core orbitals ???

mapper = ParityMapper()  # Set Mapper
print(hamiltonian)
# Do two qubit reduction
converter = QubitConverter(mapper,two_qubit_reduction=True)
reducer = TwoQubitReduction(num_particles)
qubit_op = converter.convert(hamiltonian)
qubit_op = reducer.convert(qubit_op)



