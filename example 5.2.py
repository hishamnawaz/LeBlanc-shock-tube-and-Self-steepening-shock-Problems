#Leblanc's Shock tube
import numpy as np
import matplotlib.pyplot as plt
no_of_cells= 6000
T= 5*pow(10, -6)
gamma=5/3
xleft=0
xright=1
#for the JS, LLF
cfl1=0.4
#for SW, FVS
cfl2=0.1
dx=(xright-xleft)/no_of_cells
xcenter = np.linspace(xleft + 0.5 * dx, xright - 0.5 * dx, no_of_cells)
v=np.zeros(no_of_cells)
rho_intial=np.where(xcenter<0.5, 2, pow(10, -3))
p_initial=np.where(xcenter<0.5, pow(10, 9), 1)
def primitive_to_conserved(rho, v, p):
    rhov=rho * v
    E= p/(gamma-1.0) + 0.5*rho*v**2
    return np.vstack([rho, rhov, E])
U=primitive_to_conserved(rho_intial, v, p_initial)

def conserved_to_primitive(U, eps_rho=pow(10, -12), eps_p=pow(10, -12)):
    rho = np.maximum(U[0], eps_rho)
    v = U[1] / rho
    e_kin = 0.5 * rho * v**2
    p = (gamma - 1.0) * np.maximum(U[2] - e_kin, eps_p)
    return rho, v, p

def fluxcal(U):
    rho, v, p = conserved_to_primitive(U)
    f1 = U[1]
    f2 = U[1] * v + p
    f3 = (U[2] + p) * v
    return np.vstack([f1, f2, f3])
def minmod(a, b):
  """Minmod limiter for slope reconstruction."""
  return np.where(
      a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0
  )

def shock_sensor(rho, p):
  """Shock-sensor based on pressure gradient to detect strong compression."""
  dp = np.abs(np.gradient(p))
  p_avg = np.maximum(p, 1e-10)
  sensor = dp / p_avg
  return np.clip(sensor / (sensor + 0.5), 0.0, 1.0)

def llf_flux(U_L, U_R):
  """Local Lax-Friedrichs Riemann solver."""
  rho_L, v_L, p_L = conserved_to_primitive(U_L)
  rho_R, v_R, p_R = conserved_to_primitive(U_R)
  c_L = np.sqrt(gamma * p_L / rho_L)
  c_R = np.sqrt(gamma * p_R / rho_R)
  alpha = np.maximum(np.abs(v_L) + c_L, np.abs(v_R) + c_R)
  F_L = fluxcal(U_L)
  F_R = fluxcal(U_R)
  return 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)


def compute_rhs(U):
  """Computes spatial residual -dF/dx with ghost cells and BP limiting."""
  # Outflow boundary conditions using ghost cells
  U_ext = np.hstack([U[:, :1], U, U[:, -1:]])
  rho_ext, v_ext, p_ext = conserved_to_primitive(U_ext)

  # Compute slopes on expanded domain
  dU_left = U_ext[:, 1:-1] - U_ext[:, :-2]
  dU_right = U_ext[:, 2:] - U_ext[:, 1:-1]

  dU_cent = 0.5 * (dU_left + dU_right)
  dU_lim = minmod(dU_left, dU_right)

  sensor = shock_sensor(rho_ext, p_ext)[1:-1]
  dU = (1.0 - sensor) * dU_cent + sensor * dU_lim
  dU_padded = np.pad(dU, ((0, 0), (1, 1)), mode="edge")

  # Reconstructed states at cell interfaces
  U_left_face = U_ext[:, :-1] + 0.5 * dU_padded[:, :-1]
  U_right_face = U_ext[:, 1:] - 0.5 * dU_padded[:, 1:]

  # Bound preservation (positivity enforcement)
  rho_L, v_L, p_L = conserved_to_primitive(U_left_face)
  rho_R, v_R, p_R = conserved_to_primitive(U_right_face)
  U_left_face = primitive_to_conserved(rho_L, v_L, p_L)
  U_right_face = primitive_to_conserved(rho_R, v_R, p_R)

  # Compute LLF fluxes across all interfaces
  F_interfaces = llf_flux(U_left_face, U_right_face)

  # Flux divergence for real cells
  dUdt = -(F_interfaces[:, 1:] - F_interfaces[:, :-1]) / dx
  return dUdt



t = 0.0
step = 0

print("Starting LeBlanc shock tube simulation...")
while t < T:
  rho, v_arr, p = conserved_to_primitive(U)
  c = np.sqrt(gamma * p / rho)
  max_wave_speed = np.max(np.abs(v_arr) + c)

  dt = cfl1 * dx / max_wave_speed
  if t + dt > T:
    dt = T - t

  # 3rd-order Strong Stability Preserving Runge-Kutta (SSP-RK3)
  U1 = U + dt * compute_rhs(U)
  U2 = 0.75 * U + 0.25 * (U1 + dt * compute_rhs(U1))
  U = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * compute_rhs(U2))

  t += dt
  step += 1

  if step % 500 == 0 or t >= T:
    print(
        f"Step {step:5d} | Time: {t:.4e} / {T:.4e} s | max(rho):"
        f" {np.max(rho):.4f}"
    )

rho_final, v_final, p_final = conserved_to_primitive(U)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(xcenter, rho_final, "b-", label="Density (ρ)")
axes[0].set_title("Density Profile (ρ)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("ρ")
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(xcenter, v_final, "r-", label="Velocity (v)")
axes[1].set_title("Velocity Profile (v)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("v")
axes[1].grid(True, linestyle="--", alpha=0.6)

axes[2].semilogy(xcenter, p_final, "g-", label="Pressure (p)")
axes[2].set_title("Log-Pressure Profile (p)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("p")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.suptitle(
    f"LeBlanc Shock Tube Solution at T = {T:.1e} (N={no_of_cells} cells)",
    fontsize=14,
)
plt.tight_layout()
plt.show()