#from Quantech.verlet_integrator import distance
from force import calc_forces
from force import calc_cov
import numpy as np

#implementation of the explicit Euler integrator
femto = 1.0e-15
angst = 1.0e-10

def distance(coordinates1,coordinates2):
    return np.sqrt(np.power((coordinates1[0] - coordinates2[0]),2) + np.power((coordinates1[1] - coordinates2[1]),2) + np.power((coordinates1[2] - coordinates2[2]),2))

#mass of particle1 (H)
mass_0 = 1.67e-27
#mass of particle2 (H)
mass_1 = 1.67e-27
#time step of integrator
dt = 0.2*femto
#initial velocity of atoms
v_init_0 = np.array([0,0,0])
v_init_1 = np.array([0,0,0])

v_init = np.concatenate((v_init_0,v_init_1),axis=None)

#initial positions of atoms
init_pos = [np.array([0,0,0]),np.array([1,1,1])/np.sqrt(3)]

#integrator timesteps
times = np.arange(0*femto, 30.0*femto, dt)
#coordinate array
coords = [init_pos]
force = []

#temperature in ?? 
T = 1

m = 10 #times which we calculate the force
#
#hold v_next
v_next = None

for time in times:
    r = coords[-1]
    (f0,f1) = calc_forces(coords[-1])
    COV = calc_cov(r)
    
    #calculation of force covariance matrix
     
    if time == 0:

        v_next = -1/(2*T)*np.matmul(COV,v_init) + dt*np.concatenate((f0/mass_0,f1/mass_1),axis=None)
        
        r_next = (np.concatenate((r[0],r[1]),axis=0)*angst + dt*v_next)/angst

        r_split = np.array_split(r_next,2)

        coords.append(r_split)

    else:

        v_next = -1/(2*T)*np.matmul(COV,v_next) + dt*np.concatenate((f0/mass_0,f1/mass_1),axis=None)

        r_next = (np.concatenate((r[0],r[1]),axis=None)*angst + dt*v_next)/angst

        r_split = np.array_split(r_next,2)

        coords.append(r_split)

    print(distance(coords[-1][0],coords[-1][1]))  
