import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso

# Rosenbrock function
def f(x):
    f = (1.-x[0])**2 + (x[1]-x[0]**2)**2
    return f

# Define bounds on variable
lb = [-5., -5.]
ub = [5., 5.]

# Instantiation of PSO
mypb = pso.Pso(maxiter=100, verbose=True, particle_output=True)

# Initialize the problem
mypb.initialize(f, lb, ub)

# Optimize in parallel
swarmsize = 50
res = mypb.optimize(swarmsize=swarmsize, processes=4)

# Save the full convergence phase on disk if necessary
itera = int(res['convergence functions'].shape[0] / swarmsize)
convX = res['convergence particles']
convF = res['convergence functions'].reshape(itera*swarmsize,1)
convI = np.zeros((itera*swarmsize,1))
for i in range(0, itera+1):
		convI[i*swarmsize:i*swarmsize+swarmsize,0] = i
convPSO = np.hstack((convX, convF, convI))
#np.savetxt('convPSO.txt', convPSO)

# Plot convergence of particles
plt.ion()
for i in range(0, itera):
    plt.clf()
    plt.axis([-0.1,5.1,-0.1,5.1])
    plt.title('Iteration '+ str(i+1))
    plt.plot(res['convergence particles'][i*swarmsize:(i+1)*swarmsize,0], res['convergence particles'][i*swarmsize:(i+1)*swarmsize,1], '*', markersize=10)
    plt.pause(0.5)

