import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def oblique_delta(M,beta):
    n = (M**2)*(np.sin(beta)**2)-1
    d = (M**2)*(1.4+np.cos(2*beta))+2
    tan = (2/np.tan(beta))*(n/d)
    delta = np.arctan(tan)
    return delta

def mach_angle(M):
    return np.arcsin(1/M)


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


def taylor_maccoll(t,y):
    y1,y2 = y
    a = (1.4-1)/2
    n = (y1*(y2**2))-(a*(1-(y1**2)-(y2**2))*(2*y1+y2*(1/np.tan(t))))
    d = a*(1-(y1**2)-(y2**2))-(y2**2)
    dy2dt = n/d
    return y2, dy2dt

def wall(theta,y): return y[1]
wall.terminal = True
wall.direction = 1


def compute_theta0(beta, Mach):
    """Compute theta0 for a given beta and Mach number"""
    delta = oblique_delta(Mach, beta)
    Mach2 = M2(Mach, beta)
    Vinitial = component(find_V(Mach2), beta, delta)
    sol = solve_ivp(taylor_maccoll, t_span=[beta, 0.01], y0=Vinitial, 
                    events=wall, rtol=1e-8, atol=1e-10)
    return sol.t[-1]


def find_beta_bisection(Mach, beta_min, beta_max, tol=1e-5, max_iter=100):
    """
    Use bisection method to find beta where delta(beta) = theta0(beta)
    """
    # Function to find root of: f(beta) = delta(beta) - theta0(beta) = 0
    def f(beta):
        delta = oblique_delta(Mach, beta)
        theta0 = compute_theta0(beta, Mach)
        return delta - theta0
    
    # Check that we have opposite signs at bounds
    f_min = f(beta_min)
    f_max = f(beta_max)
    
    if f_min * f_max > 0:
        raise ValueError("No root in the given interval - f(beta_min) and f(beta_max) must have opposite signs")
    
    for iteration in range(max_iter):
        beta_mid = (beta_min + beta_max) / 2
        f_mid = f(beta_mid)
        
        if abs(f_mid) < tol:
            return beta_mid
        
        if f_min * f_mid < 0:
            beta_max = beta_mid
            f_max = f_mid
        else:
            beta_min = beta_mid
            f_min = f_mid
    
    return (beta_min + beta_max) / 2  # Return best estimate


Mach = 2

# Initial guess for beta range (in radians)
beta_min = np.radians(20)  # Lower bound
beta_max = np.radians(60)  # Upper bound

try:
    beta_solution = find_beta_bisection(Mach, beta_min, beta_max)
    delta_solution = oblique_delta(Mach, beta_solution)
    theta0_solution = compute_theta0(beta_solution, Mach)
    
    print(f"Solution found:")
    print(f"beta = {np.degrees(beta_solution):.6f} degrees")
    print(f"delta = {np.degrees(delta_solution):.6f} degrees")
    print(f"theta0 = {np.degrees(theta0_solution):.6f} degrees")
    print(f"Difference: {abs(delta_solution - theta0_solution):.2e}")
    
except ValueError as e:
    print(f"Error: {e}")
    
