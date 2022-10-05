from force import calc_forces
import numpy as np

#implementation of the simple Verlet integrator
femto = 1e-15
angst = 1e-10


#mass of particle1 (H)
mass_0 = 7*1.67e-27
#mass of particle2 (H)
mass_1 = 1.67e-27
#time step of integrator
dt = 0.2*femto
#initial velocity of atoms
v_init_0 = 0
v_init_1 = 0

#initial positions of atoms
init_pos = [0,1]

#integrator timesteps
times = np.arange(0*femto, 4.0*femto, dt)
#coordinate array
coords = [init_pos]

for time in times:
    r = coords[-1]
    (f0,f1) = calc_forces(coords[-1])
    if time == 0:
        r0_next = r[0]*angst + v_init_0*dt + 0.5*f0/mass_0*(dt*dt)
        r1_next = r[1]*angst + v_init_1*dt + 0.5*f1/mass_1*(dt*dt)
        coords.append([r0_next/angst,r1_next/angst])
    else:
        r_prev = coords[-2]
        r0_next = 2*r[0]*angst - r_prev[0]*angst + f0/mass_0*(dt*dt)
        r1_next = 2*r[1]*angst - r_prev[1]*angst + f1/mass_1*(dt*dt)
        coords.append([r0_next/angst,r1_next/angst])
    print(coords)

