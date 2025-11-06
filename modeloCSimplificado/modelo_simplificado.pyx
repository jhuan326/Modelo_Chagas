import numpy as np
cimport numpy as np
from libc.math cimport fmax 

cpdef modelo_simplificado_cy(np.ndarray[double, ndim=1] y, double t, dict p):
    
    cdef double T, I, H, SI
    cdef np.ndarray[double, ndim=1] dydt = np.zeros(4, dtype=np.double)
    
    T = fmax(0.0, y[0])
    I = fmax(0.0, y[1])
    H = fmax(0.0, y[2])
    SI = fmax(0.0, y[3])

    #dT
    dydt[0] = (p['pi_release'] * I) - (p['beta_inf'] * T * H) \
              - (p['beta_SI'] - p['evasao']) * T * SI \
              - (p['delta_T'] * T)

    #dI
    dydt[1] = (p['beta_inf'] * T * H) - (p['beta_SI_2'] * I * SI) \
              - (p['delta_I'] * I)

    #dH
    dydt[2] = (p['alfa_H'] * (p['H0'] - H)) - (p['beta_inf'] * T * H)

    #dSI
    dydt[3] = (p['alfa_SI'] * (p['SI0'] - SI)) \
              + (p['beta'] * (p['c'] * SI * T) / (p['c2'] + SI * T)) \
              - (p['delta_SI'] * SI)

    return dydt

