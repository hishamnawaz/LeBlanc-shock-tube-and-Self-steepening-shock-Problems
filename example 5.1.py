#Self steepening shock
import numpy as np
import matplotlib.pyplot as plt
xleft=-1
xright=1
t=0.0
T=0.5
no_of_cells=200
cfl=0.2
dx=(xright-xleft)/no_of_cells
xcenter=np.linspace(xleft+0.5*dx, xright-0.5*dx, no_of_cells)
xnode=np.linspace(xleft, xright, no_of_cells+1)
def intial_conditions(x):
    return np.where(abs(x)<0.2, 2, -1 )
ubar= intial_conditions(xcenter)
uperiodic= intial_conditions(xnode)
def fluxeqn(uvalue):
    return 0.5*(uvalue**2)
while t<T:
    maxv=np.max(abs(uperiodic))
    dt=cfl*dx/maxv
    if t+dt>T:
        dt=T-t
    fluxl=fluxeqn(uperiodic)
    fluxr=fluxeqn(np.roll(uperiodic, -1))
    #Bound preservation with lax fredrichs
    #alpha=np.maximum(np.abs(uperiodic), np.abs(np.roll(uperiodic, -1)))
    #fluxint=(0.5 * (fluxl + fluxr) - 0.5 * alpha * (np.roll(uperiodic, -1) - uperiodic))
    #ubar=ubar-(dt/dx)* (fluxint[1:]- fluxint[:-1])
    a=uperiodic
    a_plus=np.maximum(a, 0.0)
    a_minus=np.minimum(a, 0.0)
    du_pts_dt = - (a_plus * (uperiodic - np.roll(uperiodic, 1)) / dx + a_minus * (np.roll(uperiodic, -1) - uperiodic) / dx)
    uperiodic=uperiodic+dt*du_pts_dt
    uperiodic[-1]= uperiodic[0]
    t+=dt
plt.figure(figsize=(10, 6))
plt.plot(
    xcenter,
    intial_conditions(xcenter),
    "k:",
    alpha=0.6,
    label="Initial Wave ($u_0$)",
)
plt.plot(
    xcenter, ubar, "ro--", markersize=4, label="Cell Averages ($\\bar{u}$)"
)
plt.plot(
    xnode,
    uperiodic,
    "bx",
    markersize=5,
    label="Point Values ($u_{point}$)",
)

plt.title(
    f"Active Flux Method: Burgers' Shock (Example 5.1)\n$T = {T}$, $N = {no_of_cells}$,"
    f" CFL = {cfl}"
)
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()