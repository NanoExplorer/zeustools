import numba as nb
import numpy as np

# from MothNik on Github:
# https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4?permalink_comment_id=3961461#gistcomment-3961461


# Define Functions Using Numba
# Idea here is to solve ax = b, using least squares, where a represents our coefficients e.g. x**2, x, constants
@nb.njit(cache=True)
def _coeff_mat(x, deg):

    mat_ = np.ones(shape=(x.shape[0],deg + 1))
    mat_[:, 1] = x

    if deg > 1:
        for n in range(2, deg + 1):
            # here, the pow()-function was turned into multiplication, which gave some speedup for me (up to factor 2 for small degrees, ...
            # ... which is the normal application case)
            mat_[:, n] = mat_[:, n-1] * x

    # evaluation of the norm of each column by means of a loop
    scale_vect = np.empty((deg + 1, ))
    for n in range(0, deg + 1):
        
        # evaluation of the column's norm (stored for later processing)
        col_norm = np.linalg.norm(mat_[:, n])
        scale_vect[n] = col_norm
        
        # scaling the column to unit-length
        mat_[:, n] /= col_norm

    return mat_, scale_vect
    

@nb.jit(cache=True)
def _fit_x(a, b, scales):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    # due to the stabilization, the coefficients have the wrong scale, which is corrected now
    det_ /= scales

    return det_


@nb.jit(cache=True)
def fit_poly(x, y, deg):
    a, scales_ = _coeff_mat(x, deg)
    p = _fit_x(a, y, scales_)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]