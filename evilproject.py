import numpy as np
import scipy

def oblique_delta(M,theta,gamma):
    n = (M**2)*(np.sin(theta)**2)-1
    d = (M**2)*(gamma+np.cos(2*theta))+2
    tan = (2/np.tan(theta))*(n/d)
    delta = np.arctan(tan)
    return delta

gamma = 1.4
M = 2
theta = np.radians(50)

print(np.degrees(oblique_delta(M,theta,gamma)))


def M2(M):
    return ((M**2+5)/(7*(M**2)-1))**(1/2)

print(M2(3))