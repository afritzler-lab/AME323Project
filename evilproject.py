import numpy as np
import scipy
import matplotlib.pyplot as plt


def oblique_delta(M,beta):
    n = (M**2)*(np.sin(beta)**2)-1
    d = (M**2)*(1.4+np.cos(2*beta))+2
    tan = (2/np.tan(beta))*(n/d)
    delta = np.arctan(tan)
    return delta

def mach_angle(M):
    return 1/np.sin(1/M)


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


def taylor_maccoll(y1,y2,t):
    a = (1.4-1)/2
    n = (y1*(y2**2))-(a*(1-(y1**2)-(y2**2))*(2*y1+y2*(1/np.tan(t))))
    d = a*(1-(y1**2)-(y2**2))-(y2**2)
    dy2dt = n/d
    return dy2dt


