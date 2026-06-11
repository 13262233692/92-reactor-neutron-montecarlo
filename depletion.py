import numpy as np
from numba import njit

N_NUCLIDES = 8

i_U235 = 0
i_U238 = 1
i_Np239 = 2
i_Pu239 = 3
i_I135 = 4
i_Xe135 = 5
i_Cs137 = 6
i_Sm149 = 7

NUCLIDE_NAMES = [
    'U-235', 'U-238', 'Np-239', 'Pu-239',
    'I-135', 'Xe-135', 'Cs-137', 'Sm-149'
]

LAMBDA = np.zeros(N_NUCLIDES, dtype=np.float64)
LAMBDA[i_Np239] = 3.40785e-6
LAMBDA[i_I135] = 2.875e-5
LAMBDA[i_Xe135] = 2.094e-5
LAMBDA[i_Cs137] = 7.3e-10

SIGMA_FISSION = np.zeros(N_NUCLIDES, dtype=np.float64)
SIGMA_FISSION[i_U235] = 580.0
SIGMA_FISSION[i_Pu239] = 742.0

SIGMA_CAPTURE = np.zeros(N_NUCLIDES, dtype=np.float64)
SIGMA_CAPTURE[i_U235] = 100.0
SIGMA_CAPTURE[i_U238] = 2.7
SIGMA_CAPTURE[i_Pu239] = 271.0
SIGMA_CAPTURE[i_Xe135] = 2.65e6
SIGMA_CAPTURE[i_Sm149] = 7.42e4

BARN = 1e-24
SEC_PER_DAY = 86400.0

INITIAL_DENSITY = np.zeros(N_NUCLIDES, dtype=np.float64)
INITIAL_DENSITY[i_U235] = 0.0223
INITIAL_DENSITY[i_U238] = 0.8600

FY_I135 = np.zeros(N_NUCLIDES, dtype=np.float64)
FY_I135[i_U235] = 0.0633
FY_I135[i_Pu239] = 0.061

FY_Xe135_DIRECT = np.zeros(N_NUCLIDES, dtype=np.float64)
FY_Xe135_DIRECT[i_U235] = 0.003
FY_Xe135_DIRECT[i_Pu239] = 0.003

FY_Cs137 = np.zeros(N_NUCLIDES, dtype=np.float64)
FY_Cs137[i_U235] = 0.062
FY_Cs137[i_Pu239] = 0.065

FY_Sm149 = np.zeros(N_NUCLIDES, dtype=np.float64)
FY_Sm149[i_U235] = 0.011
FY_Sm149[i_Pu239] = 0.013


@njit(cache=True)
def _gauss_solve_8x8(A, b, n):
    aug = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            aug[i, j] = A[i, j]
        aug[i, n] = b[i]

    for col in range(n):
        pivot_row = col
        pivot_abs = abs(aug[col, col])
        for r in range(col + 1, n):
            if abs(aug[r, col]) > pivot_abs:
                pivot_abs = abs(aug[r, col])
                pivot_row = r

        if pivot_row != col:
            for c in range(n + 1):
                tmp = aug[col, c]
                aug[col, c] = aug[pivot_row, c]
                aug[pivot_row, c] = tmp

        p = aug[col, col]
        if abs(p) < 1e-40:
            for i in range(n):
                b[i] = 0.0
            return

        for c in range(col, n + 1):
            aug[col, c] /= p

        for r in range(n):
            if r == col:
                continue
            factor = aug[r, col]
            if abs(factor) < 1e-40:
                continue
            for c in range(col, n + 1):
                aug[r, c] -= factor * aug[col, c]

    for i in range(n):
        val = aug[i, n]
        if val != val:
            b[i] = 0.0
        elif val < 0:
            b[i] = 0.0
        else:
            b[i] = val


@njit(cache=True)
def _build_matrix(A, flux, dt, density):
    n = N_NUCLIDES
    sigma_a = np.zeros(n, dtype=np.float64)
    sigma_f = np.zeros(n, dtype=np.float64)
    lam = np.zeros(n, dtype=np.float64)

    sigma_f[i_U235] = SIGMA_FISSION[i_U235]
    sigma_f[i_Pu239] = SIGMA_FISSION[i_Pu239]
    sigma_a[i_U235] = SIGMA_CAPTURE[i_U235] + sigma_f[i_U235]
    sigma_a[i_U238] = SIGMA_CAPTURE[i_U238]
    sigma_a[i_Pu239] = SIGMA_CAPTURE[i_Pu239] + sigma_f[i_Pu239]
    sigma_a[i_Xe135] = SIGMA_CAPTURE[i_Xe135]
    sigma_a[i_Sm149] = SIGMA_CAPTURE[i_Sm149]

    lam[i_Np239] = LAMBDA[i_Np239]
    lam[i_I135] = LAMBDA[i_I135]
    lam[i_Xe135] = LAMBDA[i_Xe135]
    lam[i_Cs137] = LAMBDA[i_Cs137]

    for i in range(n):
        for j in range(n):
            A[i, j] = 0.0

    barn_flux = flux * BARN

    loss_U235 = barn_flux * sigma_a[i_U235]
    A[i_U235, i_U235] = -loss_U235

    loss_U238 = barn_flux * sigma_a[i_U238]
    A[i_U238, i_U238] = -loss_U238
    prod_Np239_from_U238 = barn_flux * SIGMA_CAPTURE[i_U238]
    A[i_Np239, i_U238] = prod_Np239_from_U238

    loss_Np239 = barn_flux * sigma_a[i_Np239] + lam[i_Np239]
    A[i_Np239, i_Np239] = -loss_Np239
    A[i_Pu239, i_Np239] = lam[i_Np239]

    loss_Pu239 = barn_flux * sigma_a[i_Pu239]
    A[i_Pu239, i_Pu239] = -loss_Pu239

    fission_rate = (barn_flux * sigma_f[i_U235] * density[i_U235] +
                   barn_flux * sigma_f[i_Pu239] * density[i_Pu239])

    loss_I135 = barn_flux * sigma_a[i_I135] + lam[i_I135]
    A[i_I135, i_I135] = -loss_I135

    loss_Xe135 = barn_flux * sigma_a[i_Xe135] + lam[i_Xe135]
    A[i_Xe135, i_Xe135] = -loss_Xe135
    A[i_Xe135, i_I135] = lam[i_I135]

    loss_Cs137 = barn_flux * sigma_a[i_Cs137] + lam[i_Cs137]
    A[i_Cs137, i_Cs137] = -loss_Cs137

    loss_Sm149 = barn_flux * sigma_a[i_Sm149]
    A[i_Sm149, i_Sm149] = -loss_Sm149


@njit(cache=True)
def _build_source_vec(src, flux, density):
    n = N_NUCLIDES
    sigma_f = np.zeros(n, dtype=np.float64)
    sigma_f[i_U235] = SIGMA_FISSION[i_U235]
    sigma_f[i_Pu239] = SIGMA_FISSION[i_Pu239]
    barn_flux = flux * BARN

    fission_rate_U235 = barn_flux * sigma_f[i_U235] * density[i_U235]
    fission_rate_Pu239 = barn_flux * sigma_f[i_Pu239] * density[i_Pu239]

    src[i_U235] = density[i_U235]
    src[i_U238] = density[i_U238]
    src[i_Np239] = density[i_Np239]
    src[i_Pu239] = density[i_Pu239]
    src[i_I135] = density[i_I135] + (FY_I135[i_U235] * fission_rate_U235 + FY_I135[i_Pu239] * fission_rate_Pu239)
    src[i_Xe135] = density[i_Xe135] + (FY_Xe135_DIRECT[i_U235] * fission_rate_U235 + FY_Xe135_DIRECT[i_Pu239] * fission_rate_Pu239)
    src[i_Cs137] = density[i_Cs137] + (FY_Cs137[i_U235] * fission_rate_U235 + FY_Cs137[i_Pu239] * fission_rate_Pu239)
    src[i_Sm149] = density[i_Sm149] + (FY_Sm149[i_U235] * fission_rate_U235 + FY_Sm149[i_Pu239] * fission_rate_Pu239)


@njit(cache=True)
def solve_bateman_step(flux, density_vec, dt_seconds):
    n = N_NUCLIDES
    A = np.zeros((n, n), dtype=np.float64)
    src = np.zeros(n, dtype=np.float64)

    _build_matrix(A, flux, dt_seconds, density_vec)
    _build_source_vec(src, flux, density_vec)

    for i in range(n):
        A[i, i] = A[i, i] - 1.0 / dt_seconds

    for i in range(n):
        src[i] = -src[i] / dt_seconds

    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rhs[i] = src[i]

    _gauss_solve_8x8(A, rhs, n)

    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v = rhs[i]
        if v != v or v < 0:
            out[i] = density_vec[i] * 0.5
        else:
            out[i] = v
    return out


@njit(cache=True)
def build_depletion_matrix(A, flux, density_vec):
    n = N_NUCLIDES
    _build_matrix(A, flux, 1.0, density_vec)
