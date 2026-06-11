import numpy as np
from numba import njit

CORE_RADIUS = 170.0
CORE_HEIGHT = 370.0
PITCH = 1.26
FUEL_ROD_RADIUS = 0.41
GUIDE_TUBE_RADIUS = 0.48
ASSEMBLY_SIZE = 17
N_ASSEMBLIES_X = 15
N_ASSEMBLIES_Y = 15

MAT_FUEL = 0
MAT_MODERATOR = 1
MAT_COOLANT = 2
MAT_GUIDE_TUBE = 3

U235_NUMBER_DENSITY = 0.0223
C12_NUMBER_DENSITY = 0.0802
H2O_NUMBER_DENSITY = 0.0334

ENERGY_GROUPS = np.array([
    1e-5, 6.25e-3, 0.625, 5.53e3, 8.21e3, 1.0e4,
    1.5e4, 2.0e4, 5.0e4, 1.0e5, 5.0e5, 1.0e6,
    2.0e6, 5.0e6, 1.0e7, 1.4e7, 2.0e7
], dtype=np.float64)

N_ENERGY_GROUPS = len(ENERGY_GROUPS)

U235_FISSION_XS = np.array([
    580.0, 95.0, 35.0, 3.5, 1.8, 1.2,
    0.9, 0.7, 0.45, 0.35, 1.15, 1.25,
    1.30, 1.25, 1.20, 1.18, 1.22
], dtype=np.float64)

U235_CAPTURE_XS = np.array([
    100.0, 28.0, 12.0, 6.5, 4.0, 2.8,
    2.0, 1.5, 0.6, 0.35, 0.10, 0.09,
    0.08, 0.08, 0.09, 0.09, 0.09
], dtype=np.float64)

U235_SCATTER_XS = np.array([
    12.0, 10.0, 8.0, 5.5, 5.0, 4.8,
    4.5, 4.2, 4.0, 4.0, 4.5, 5.0,
    5.5, 6.0, 5.5, 5.0, 5.5
], dtype=np.float64)

C12_SCATTER_XS = np.array([
    5.0, 4.8, 4.7, 4.6, 4.5, 4.5,
    4.5, 4.5, 4.6, 4.7, 4.8, 4.9,
    5.0, 5.0, 4.5, 4.0, 3.5
], dtype=np.float64)

C12_CAPTURE_XS = np.array([
    0.004, 0.003, 0.002, 0.0015, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
    0.001, 0.001, 0.001, 0.001, 0.001
], dtype=np.float64)

H1_SCATTER_XS = np.array([
    80.0, 30.0, 20.0, 10.0, 7.0, 5.5,
    4.0, 3.5, 3.0, 2.5, 2.0, 2.0,
    2.0, 1.8, 1.5, 1.2, 1.0
], dtype=np.float64)

H1_CAPTURE_XS = np.array([
    0.33, 0.15, 0.06, 0.01, 0.005, 0.003,
    0.002, 0.001, 0.0005, 0.0003, 0.0001, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001
], dtype=np.float64)

O16_SCATTER_XS = np.array([
    4.2, 3.8, 3.5, 3.2, 3.0, 2.8,
    2.5, 2.3, 2.2, 2.1, 2.0, 2.5,
    3.0, 3.0, 2.5, 2.0, 1.5
], dtype=np.float64)

O16_CAPTURE_XS = np.array([
    0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001
], dtype=np.float64)

NU_BAR = 2.43

FISSION_ENERGY_SPECTRUM = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0001,
    0.0005, 0.001, 0.01, 0.03, 0.15, 0.20,
    0.25, 0.20, 0.10, 0.05, 0.0094
], dtype=np.float64)

FISSION_ENERGY_CDF = np.cumsum(FISSION_ENERGY_SPECTRUM)
FISSION_ENERGY_CDF /= FISSION_ENERGY_CDF[-1]


@njit(cache=True)
def get_energy_group(energy):
    idx = 0
    for i in range(N_ENERGY_GROUPS - 1):
        if energy >= ENERGY_GROUPS[i]:
            idx = i
    return idx


@njit(cache=True)
def sample_fission_energy():
    u = np.random.random()
    idx = 0
    for i in range(len(FISSION_ENERGY_CDF)):
        if u <= FISSION_ENERGY_CDF[i]:
            idx = i
            break
    else:
        idx = len(FISSION_ENERGY_CDF) - 1
    if idx == 0:
        return ENERGY_GROUPS[0] * (1.0 + np.random.random())
    e_low = ENERGY_GROUPS[max(idx - 1, 0)]
    e_high = ENERGY_GROUPS[idx]
    frac = np.random.random()
    return e_low + frac * (e_high - e_low)


@njit(cache=True)
def get_macro_xs(material, energy_group):
    if material == MAT_FUEL:
        sigma_t = U235_NUMBER_DENSITY * (
            U235_FISSION_XS[energy_group] +
            U235_CAPTURE_XS[energy_group] +
            U235_SCATTER_XS[energy_group]
        )
        sigma_f = U235_NUMBER_DENSITY * U235_FISSION_XS[energy_group]
        sigma_c = U235_NUMBER_DENSITY * U235_CAPTURE_XS[energy_group]
        sigma_s = U235_NUMBER_DENSITY * U235_SCATTER_XS[energy_group]
        return sigma_t, sigma_f, sigma_c, sigma_s
    elif material == MAT_MODERATOR:
        sigma_t = C12_NUMBER_DENSITY * (
            C12_SCATTER_XS[energy_group] + C12_CAPTURE_XS[energy_group]
        )
        sigma_f = 0.0
        sigma_c = C12_NUMBER_DENSITY * C12_CAPTURE_XS[energy_group]
        sigma_s = C12_NUMBER_DENSITY * C12_SCATTER_XS[energy_group]
        return sigma_t, sigma_f, sigma_c, sigma_s
    elif material == MAT_COOLANT:
        sigma_s_h2o = H2O_NUMBER_DENSITY * H1_SCATTER_XS[energy_group]
        sigma_c_h2o = H2O_NUMBER_DENSITY * H1_CAPTURE_XS[energy_group]
        sigma_s_o = 0.5 * H2O_NUMBER_DENSITY * O16_SCATTER_XS[energy_group]
        sigma_c_o = 0.5 * H2O_NUMBER_DENSITY * O16_CAPTURE_XS[energy_group]
        sigma_t = sigma_s_h2o + sigma_c_h2o + sigma_s_o + sigma_c_o
        sigma_f = 0.0
        sigma_c = sigma_c_h2o + sigma_c_o
        sigma_s = sigma_s_h2o + sigma_s_o
        return sigma_t, sigma_f, sigma_c, sigma_s
    else:
        sigma_t = C12_NUMBER_DENSITY * (
            C12_SCATTER_XS[energy_group] + C12_CAPTURE_XS[energy_group]
        )
        sigma_f = 0.0
        sigma_c = C12_NUMBER_DENSITY * C12_CAPTURE_XS[energy_group]
        sigma_s = C12_NUMBER_DENSITY * C12_SCATTER_XS[energy_group]
        return sigma_t, sigma_f, sigma_c, sigma_s


@njit(cache=True)
def get_A(material):
    if material == MAT_FUEL:
        return 235.0
    elif material == MAT_MODERATOR:
        return 12.0
    elif material == MAT_COOLANT:
        return 1.0
    else:
        return 12.0
