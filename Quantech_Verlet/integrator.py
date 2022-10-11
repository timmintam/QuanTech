from force import calc_forces
import numpy as np

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

#initial positions of atoms
init_pos = [np.array([0,0,0]),np.array([1,1,1])/np.sqrt(3)]

#integrator timesteps
times = np.arange(0*femto, 15.0*femto, dt)

#coordinate array
coords = [init_pos]
force = []

for time in times:
    r = coords[-1]

    (f0,f1) = calc_forces(r)

    if time == 0:

        r0_next = r[0]*angst + v_init_0*dt - 0.5*f0/mass_0*(dt*dt)
        r1_next = r[1]*angst + v_init_1*dt - 0.5*f1/mass_1*(dt*dt)

        coords.append([r0_next/angst,r1_next/angst])
    else:
        r_prev = coords[-2]

        r0_next = 2*r[0]*angst - r_prev[0]*angst - f0/mass_0*(dt*dt)
        r1_next = 2*r[1]*angst - r_prev[1]*angst - f1/mass_1*(dt*dt)

        coords.append([r0_next/angst,r1_next/angst])

    print(distance(coords[-1][0],coords[-1][1]))    
