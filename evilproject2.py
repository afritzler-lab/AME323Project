import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root_scalar
import matplotlib.pyplot as plt


def oblique_delta(M,beta):
    n = (M**2)*(np.sin(beta)**2)-1
    d = (M**2)*(1.4+np.cos(2*beta))+2
    tan = (2/np.tan(beta))*(n/d)
    delta = np.arctan(tan)
    return delta

def mach_angle(M):
    return 1/np.arcsin(1/M)


def M2(M1,beta):
    M1n = M1*np.sin(beta)
    M2n = ((M1n**2+5)/(7*(M1n**2)-1))**(1/2)
    M = M2n/(np.sin(beta-oblique_delta(M1,beta)))
    return M


def find_V(M2):
    V = ((2/(0.4*(M2**2)))+1)**(-1/2)
    return V

def component(V,beta,delta):
    Vr = V*np.cos(beta-delta)
    Vt = -V*np.sin(beta-delta)
    return Vr,Vt


def taylor_maccoll(t, y):
    y1, y2 = y
    a = (1.4 - 1) / 2

    n = (y1 * (y2**2)) - (a * (1 - (y1**2) - (y2**2)) * (2*y1 + y2*(1/np.tan(t))))
    d = a * (1 - (y1**2) - (y2**2)) - (y2**2)


    if abs(d) < 1e-8:
        dy2dt = 0.0
    else:
        dy2dt = n / d

    return [y2, dy2dt]

def residual(beta, M):
    beta = np.asarray(beta)
    if beta.shape:
        beta = float(beta[0])
    else:
        beta = float(beta)

    delta = oblique_delta(M, beta)
    M2_val = M2(M, beta)
    V = find_V(M2_val)

    Vr0, Vt0 = component(V, beta, delta)

    sol = solve_ivp(
        taylor_maccoll,
        [beta, 1e-6],
        [Vr0, Vt0],
        events=wall,
        rtol=1e-8,
        atol=1e-10
    )

    if len(sol.t_events[0]) == 0:
        return 1.0

    return sol.y_events[0][0][1]

def wall(theta,y): return y[1]
wall.terminal = True
wall.direction = 1


Mach = 2

beta_guess = np.radians(65)
beta_solution = fsolve(residual, beta_guess, args=(Mach,))

beta = beta_solution[0]

output_beta = beta



delta = oblique_delta(Mach, beta)
Mach2 = M2(Mach, beta)
V = find_V(Mach2)

Vr0, Vt0 = component(V, beta, delta)

sol = solve_ivp(
    taylor_maccoll,
    [beta, 1e-6],
    [Vr0, Vt0],
    events=wall,
    rtol=1e-8,
    atol=1e-10
)

theta_c = sol.t_events[0][0]

print(f"Shock angle beta (deg): {np.degrees(beta):.3f}")
print(f"Cone angle theta_c (deg): {np.degrees(theta_c):.3f}")


def compute_theta_c_for_M_beta(M, beta):
    delta = oblique_delta(M, beta)
    M2_val = M2(M, beta)
    V = find_V(M2_val)
    Vr0, Vt0 = component(V, beta, delta)

    sol_tmp = solve_ivp(
        taylor_maccoll,
        [beta, 1e-6],
        [Vr0, Vt0],
        events=wall,
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-2
    )

    if len(sol_tmp.t_events[0]) == 0:
        return np.nan
    return sol_tmp.t_events[0][0]


Mach_values = [1.5, 2.0, 5.0]
beta_start = np.radians(5)
beta_end = np.radians(89.5)
beta_samples = np.linspace(beta_start, beta_end, 200)

results = {}
for Mval in Mach_values:
    theta_list = []
    for b in beta_samples:
        if b <= np.arcsin(1 / Mval):
            theta_list.append(np.nan)
            continue
        theta_list.append(compute_theta_c_for_M_beta(Mval, b))
    results[Mval] = np.array(theta_list)

plt.figure(figsize=(8, 5))
for Mval in Mach_values:
    th = results[Mval]
    valid = ~np.isnan(th)
    plt.plot(np.degrees(beta_samples[valid]), np.degrees(th[valid]), label=f"Mach {Mval}")

plt.xlabel("Shock angle beta (deg)")
plt.ylabel("Cone angle theta_c (deg)")
plt.title("Beta vs Theta_c for different Mach numbers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("beta_vs_theta_c.png", dpi=150)
plt.show()
