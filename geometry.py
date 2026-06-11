import numpy as np
from numba import njit
from nuclear_data import (
    PITCH, FUEL_ROD_RADIUS, GUIDE_TUBE_RADIUS, ASSEMBLY_SIZE,
    N_ASSEMBLIES_X, N_ASSEMBLIES_Y, CORE_RADIUS, CORE_HEIGHT,
    MAT_FUEL, MAT_MODERATOR, MAT_COOLANT, MAT_GUIDE_TUBE
)

CORE_X_MIN = -(N_ASSEMBLIES_X * ASSEMBLY_SIZE * PITCH) / 2.0
CORE_X_MAX = (N_ASSEMBLIES_X * ASSEMBLY_SIZE * PITCH) / 2.0
CORE_Y_MIN = -(N_ASSEMBLIES_Y * ASSEMBLY_SIZE * PITCH) / 2.0
CORE_Y_MAX = (N_ASSEMBLIES_Y * ASSEMBLY_SIZE * PITCH) / 2.0
CORE_Z_MIN = 0.0
CORE_Z_MAX = CORE_HEIGHT

GUIDE_TUBE_LIST = [
    (2, 5), (2, 8), (2, 11),
    (3, 3), (3, 13),
    (5, 2), (5, 5), (5, 8), (5, 11), (5, 14),
    (8, 2), (8, 5), (8, 8), (8, 11), (8, 14),
    (11, 2), (11, 5), (11, 8), (11, 11), (11, 14),
    (13, 3), (13, 13),
    (14, 5), (14, 8), (14, 11),
]
GUIDE_TUBE_SET = set(GUIDE_TUBE_LIST)
GUIDE_TUBE_X = np.array([p[0] for p in GUIDE_TUBE_LIST], dtype=np.int64)
GUIDE_TUBE_Y = np.array([p[1] for p in GUIDE_TUBE_LIST], dtype=np.int64)
N_GUIDE_TUBES = len(GUIDE_TUBE_LIST)

FUEL_PIN_MASK = np.ones((ASSEMBLY_SIZE, ASSEMBLY_SIZE), dtype=bool)
for (gx, gy) in GUIDE_TUBE_LIST:
    FUEL_PIN_MASK[gx, gy] = False
FUEL_PIN_COORDS = np.argwhere(FUEL_PIN_MASK).astype(np.int64)
N_FUEL_PINS_PER_ASSEMBLY = len(FUEL_PIN_COORDS)

SOURCE_ASSEMBLY_RANGE = 3
SOURCE_CENTER_AX = N_ASSEMBLIES_X // 2
SOURCE_CENTER_AY = N_ASSEMBLIES_Y // 2

SOURCE_ASSEMBLY_OFFSETS = []
for dax in range(-(SOURCE_ASSEMBLY_RANGE // 2), SOURCE_ASSEMBLY_RANGE // 2 + 1):
    for day in range(-(SOURCE_ASSEMBLY_RANGE // 2), SOURCE_ASSEMBLY_RANGE // 2 + 1):
        ax = SOURCE_CENTER_AX + dax
        ay = SOURCE_CENTER_AY + day
        SOURCE_ASSEMBLY_OFFSETS.append((ax, ay))
SOURCE_ASSEMBLY_OFFSETS = np.array(SOURCE_ASSEMBLY_OFFSETS, dtype=np.int64)
N_SOURCE_ASSEMBLIES = len(SOURCE_ASSEMBLY_OFFSETS)
TOTAL_SOURCE_PINS = N_SOURCE_ASSEMBLIES * N_FUEL_PINS_PER_ASSEMBLY


@njit(cache=True)
def source_position_from_index(idx):
    ax, ay = SOURCE_ASSEMBLY_OFFSETS[idx // N_FUEL_PINS_PER_ASSEMBLY]
    pi, pj = FUEL_PIN_COORDS[idx % N_FUEL_PINS_PER_ASSEMBLY]
    x = CORE_X_MIN + (ax * ASSEMBLY_SIZE + pi + 0.5) * PITCH
    y = CORE_Y_MIN + (ay * ASSEMBLY_SIZE + pj + 0.5) * PITCH
    return x, y


def build_material_map(resolution_xy=2.0, resolution_z=10.0):
    nx = int((CORE_X_MAX - CORE_X_MIN) / resolution_xy)
    ny = int((CORE_Y_MAX - CORE_Y_MIN) / resolution_xy)
    nz = int((CORE_Z_MAX - CORE_Z_MIN) / resolution_z)
    mat_map = np.full((nx, ny, nz), MAT_MODERATOR, dtype=np.int8)
    x_centers = np.linspace(CORE_X_MIN + resolution_xy / 2,
                           CORE_X_MAX - resolution_xy / 2, nx)
    y_centers = np.linspace(CORE_Y_MIN + resolution_xy / 2,
                           CORE_Y_MAX - resolution_xy / 2, ny)
    core_center_x = (CORE_X_MIN + CORE_X_MAX) / 2.0
    core_center_y = (CORE_Y_MIN + CORE_Y_MAX) / 2.0
    for ix in range(nx):
        for iy in range(ny):
            dx = x_centers[ix] - core_center_x
            dy = y_centers[iy] - core_center_y
            r_from_center = np.sqrt(dx * dx + dy * dy)
            if r_from_center > CORE_RADIUS:
                continue
            loc_x = x_centers[ix] - CORE_X_MIN
            loc_y = y_centers[iy] - CORE_Y_MIN
            pin_ix = int(loc_x / PITCH)
            pin_iy = int(loc_y / PITCH)
            pin_x = CORE_X_MIN + (pin_ix + 0.5) * PITCH
            pin_y = CORE_Y_MIN + (pin_iy + 0.5) * PITCH
            dist = np.sqrt((x_centers[ix] - pin_x) ** 2 +
                          (y_centers[iy] - pin_y) ** 2)
            local_pin_x = pin_ix % ASSEMBLY_SIZE
            local_pin_y = pin_iy % ASSEMBLY_SIZE
            if (local_pin_x, local_pin_y) in GUIDE_TUBE_SET:
                if dist < GUIDE_TUBE_RADIUS:
                    mat_map[ix, iy, :] = MAT_COOLANT
                else:
                    mat_map[ix, iy, :] = MAT_MODERATOR
            else:
                if dist < FUEL_ROD_RADIUS:
                    mat_map[ix, iy, :] = MAT_FUEL
                else:
                    mat_map[ix, iy, :] = MAT_MODERATOR
    return mat_map, x_centers, y_centers


@njit(cache=True)
def get_material(x, y, z):
    if x < CORE_X_MIN or x > CORE_X_MAX:
        return -1
    if y < CORE_Y_MIN or y > CORE_Y_MAX:
        return -1
    if z < CORE_Z_MIN or z > CORE_Z_MAX:
        return -1
    core_center_x = (CORE_X_MIN + CORE_X_MAX) / 2.0
    core_center_y = (CORE_Y_MIN + CORE_Y_MAX) / 2.0
    r_from_center = np.sqrt((x - core_center_x) ** 2 +
                           (y - core_center_y) ** 2)
    if r_from_center > CORE_RADIUS:
        return -1
    loc_x = x - CORE_X_MIN
    loc_y = y - CORE_Y_MIN
    pin_ix = int(loc_x / PITCH)
    pin_iy = int(loc_y / PITCH)
    pin_x = CORE_X_MIN + (pin_ix + 0.5) * PITCH
    pin_y = CORE_Y_MIN + (pin_iy + 0.5) * PITCH
    dist = np.sqrt((x - pin_x) ** 2 + (y - pin_y) ** 2)
    local_pin_x = pin_ix % ASSEMBLY_SIZE
    local_pin_y = pin_iy % ASSEMBLY_SIZE
    is_guide = False
    for k in range(N_GUIDE_TUBES):
        if local_pin_x == GUIDE_TUBE_X[k] and local_pin_y == GUIDE_TUBE_Y[k]:
            is_guide = True
            break
    if is_guide:
        if dist < GUIDE_TUBE_RADIUS:
            return MAT_COOLANT
        else:
            return MAT_MODERATOR
    else:
        if dist < FUEL_ROD_RADIUS:
            return MAT_FUEL
        else:
            return MAT_MODERATOR


def get_fuel_rod_positions(n_assemblies=3):
    positions = []
    center_ax = N_ASSEMBLIES_X // 2
    center_ay = N_ASSEMBLIES_Y // 2
    for ax in range(center_ax - n_assemblies // 2, center_ax + n_assemblies // 2 + 1):
        for ay in range(center_ay - n_assemblies // 2, center_ay + n_assemblies // 2 + 1):
            for i in range(ASSEMBLY_SIZE):
                for j in range(ASSEMBLY_SIZE):
                    if (i, j) not in GUIDE_TUBE_SET:
                        x = CORE_X_MIN + (ax * ASSEMBLY_SIZE + i + 0.5) * PITCH
                        y = CORE_Y_MIN + (ay * ASSEMBLY_SIZE + j + 0.5) * PITCH
                        r = np.sqrt((x - (CORE_X_MIN + CORE_X_MAX) / 2) ** 2 +
                                   (y - (CORE_Y_MIN + CORE_Y_MAX) / 2) ** 2)
                        if r < CORE_RADIUS:
                            positions.append((x, y))
    return positions


def get_fuel_rod_positions_full():
    positions = []
    center_x = (CORE_X_MIN + CORE_X_MAX) / 2.0
    center_y = (CORE_Y_MIN + CORE_Y_MAX) / 2.0
    for ax in range(N_ASSEMBLIES_X):
        for ay in range(N_ASSEMBLIES_Y):
            for i in range(ASSEMBLY_SIZE):
                for j in range(ASSEMBLY_SIZE):
                    if (i, j) not in GUIDE_TUBE_SET:
                        x = CORE_X_MIN + (ax * ASSEMBLY_SIZE + i + 0.5) * PITCH
                        y = CORE_Y_MIN + (ay * ASSEMBLY_SIZE + j + 0.5) * PITCH
                        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if r < CORE_RADIUS:
                            positions.append((x, y))
    return positions
