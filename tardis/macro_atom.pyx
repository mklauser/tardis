# module for fast macro_atom calculations

# cython: profile=False
# cython: boundscheck=True
# cython: cdivision=True
# cython: wraparound=False

import numpy as np

from cython.view cimport array as cvarray
cimport numpy as np

ctypedef np.int64_t int_type_t

from astropy import constants

cdef extern from "math.h":
    double exp(double)


cdef double h_cgs = constants.h.cgs.value
cdef double c = constants.c.cgs.value
cdef double kb = constants.k_B.cgs.value
cdef double inv_c2 = 1 / (c ** 2)



# DEPRECATED doing with numpy seems to be faster.
def intensity_black_body(np.ndarray[double, ndim=1] nus, double t_rad, np.ndarray[double, ndim=1] j_nus):
    cdef double c1 = h_cgs * inv_c2
    cdef double j_nu, nu
    cdef int i

    cdef double beta_rad = 1 / (kb * t_rad)

    for i in range(len(nus)):
        nu = nus[i]

        j_nus[i] = (c1 * nu ** 3) / (exp(h_cgs * nu * beta_rad) - 1)

def calculate_beta_sobolev(np.ndarray[double, ndim=1] tau_sobolevs, np.ndarray[double, ndim=1] beta_sobolevs):
    cdef double beta_sobolev
    cdef double tau_sobolev
    cdef int i

    for i in range(len(tau_sobolevs)):
        tau_sobolev = tau_sobolevs[i]

        if tau_sobolev > 1e3:
            beta_sobolev = 1 / tau_sobolev
        elif tau_sobolev < 1e-4:
            beta_sobolev = 1 - 0.5 * tau_sobolev
        else:
            beta_sobolev = (1 - exp(-tau_sobolev)) / tau_sobolev
        beta_sobolevs[i] = beta_sobolev

def normalize_transition_probabilities(double [:, :] p_transition,
                                       int_type_t [:] reference_levels):
    cdef int i, j, k
    cdef np.ndarray[double, ndim=1] norm_factor = np.zeros(p_transition.shape[1])
    cdef int start_id = 0
    cdef  end_id = 0

    for i in range(len(reference_levels) - 1):
        norm_factor[:] = 0.0
        for j in range(reference_levels[i], reference_levels[i + 1]):
            for k in range(0, p_transition.shape[1]):
                norm_factor[k] += p_transition[j, k]
        for j in range(reference_levels[i], reference_levels[i + 1]):
            for k in range(0, p_transition.shape[1]):
                if norm_factor[k] == 0.0:
                    continue

                p_transition[j,k] /= norm_factor[k]


cpdef calculate_collisional_deexcitation_rate( int number_of_shells,
                                            double [:] electron_densities,
                                            double [:,:,:] interpolated_collision_a,
                                            double [:] wavelength_cm,
                                            double [:,:] detailed_balance,
                                            double [:,:,:] collisional_deexcitation_rates
                                            ) nogil:
    """
    Returns the calculate collisional deexcitation rate for the macroatom
    @param electron_densities: 1-d array like data storing the electron densities for each shell
    @param interpolated_collision_a: 2-d array like data storing the interpolated_collision_a for each shell, start-level, and end-level
    @param wavelength_cm: 1-d array like data storing the wavelength in cm for each level
    @param level_populations_lte: 2-d array like data storing the level populations in lte for each shell and level
    @return: collisional deexcitation rates: 3-d array like data storing for each shell, start level, and end level.
    We use the pointer to collisional_deexcitation_rates for return
    """
    cdef int  sl, el, nl , s, ns # sl start-level; el end-level; nl number of levels; s shell; ns number of shells;
    cdef double [:,:] db # detailed balance factor for nlte inversion
    ns = number_of_shells
    nl = len(detailed_balance)
    db = detailed_balance

    for s in range(ns):
        #over all shell

        for sl in range(nl):
            for el in range(nl):
                #over all levels
                collisional_deexcitation_rates[s,sl,el] = db[s] * electron_densities[s] * interpolated_collision_a[s, sl, el] *  h_cgs * c / wavelength_cm[sl]

cpdef calculate_collisional_recombination_rate( int number_of_shells,
                                                double [:,:] level_populations_lte,
                                                double [:,:] collisional_ionization_rate
                                                ) nogil:
    """
    Returns the calculate collisional recombination rate computed from the collisional ionization rates. 
    """
    cdef int  sl, el, nl , s, ns # sl start-level; el end-level; nl number of levels; s shell; ns number of shells;
    cdef double db # detailed balance factor for nlte inversion
    ns = number_of_shells
    nl = len(level_populations_lte)

    for s in range(ns):
        for sl in range(nl):
            for el in range(nl):
                db = (level_populations_lte[s,sl] / level_populations_lte[s,el])


                                                



