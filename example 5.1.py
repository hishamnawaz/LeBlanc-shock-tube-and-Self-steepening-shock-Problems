import numpy as np
import matplotlib.pyplot as plt
#We initialize the Variables as mentioned in the Problem
xleft=-1
xright=1
T=0.5
t=0.0
cfl=0.2
n_cells=200
dx=(xright-xleft)/n_cells
xcenter=np.linspace(xleft+0.5*dx, xright-0.5*dx, n_cells)
xnode=np.linspace(xleft, xright, n_cells+1)
#We use the intial conditions mentioned in the problem 
def initial_conditions(x):
    return np.where(abs(x)< 0.2, 2, -1)
ubar=initial_conditions(xcenter)
uperiodic=initial_conditions(xnode)
#We define a FLux function to calculate flux from the given problem
def flux_equation(u):
    return 0.5*(u**2)
while t<T:
    max_velocity=np.max(uperiodic)
    dt=cfl*dx/max_velocity
    if t+dt>T:
        dt=T-t
    flux_left=flux_equation(uperiodic)
    flux_right=flux_equation(np.roll(uperiodic, -1))
# The Bound preservation by using Lax Freidrichs
#COmment out alpha, fluxint, ubar to switch between bound preservations
    alpha=np.maximum(np.abs(uperiodic), np.abs(np.roll(uperiodic, -1)))
    flux_int= (0.5*(flux_left+flux_right)) - (0.5*alpha*uperiodic)
    ubar=ubar- (dt/dx)* (flux_int[1:]- flux_int[:-1])
    a=uperiodic
    a_plus=np.maximum(a, 0.0)
    a_minus=np.minimum(a, 0.0)
    du_pts_dt = - (a_plus * (uperiodic - np.roll(uperiodic, 1)) / dx + a_minus * (np.roll(uperiodic, -1) - uperiodic) / dx)
    uperiodic=uperiodic+dt*du_pts_dt
    uperiodic[-1]= uperiodic[0]
    t+=dt

# We display the plot in this section 
plt.figure(figsize=(9,6))
plt.plot(xcenter, initial_conditions(xcenter), "k:", alpha=0.6, label="Initial Wave  ($u_0$)")
plt.plot(xcenter, ubar, "ro--", markersize=4, label="Cell Averages ($\\bar{u}$)")
plt.plot(xnode, uperiodic, "bx", markersize=5, label="Point Values ($u_{point}$)")
plt.title(f" Active Flux Methods (Example : 5.1), \n$T = {T}$, $N = {n_cells}$,"  f" CFL = {cfl}")
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
