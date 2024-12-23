# The workhorse of the project.
# Everything in one place from defining FS & L to computing sigma.
#
# Uses depth-first recursion for time ordered integration,
#   a trick invented for Boltzmann transport theory by Roemer Hinlopen (author)
#
# 04-2024 Start coding
# Author: Roemer Hinlopen

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
from numba import njit, i8, f8
import time

# See Buckley1991
# b-axis is the direction of the chain as per usual convention
A = 3.821e-10
B = 3.883e-10
C = 11.69e-10
HBAR = 1.054571817e-34
Q = 1.60217646e-19
_st  = time.process_time()

# Note: Recursion and numba caching currently is bugged
#   and so a lot has to be recompiled each execution.

#############################
# Initialise fs
#############################


def _import_fs_data_file(name):
    """ Names are string. 'chain', 'inner_square', 'outer_square' or 'S' """

    if name == 'iso_2d' or name == 'iso_1d':
        return np.array([[], []])
    if not name in ['chain', 'inner_square', 'outer_square', 'S']:
        raise ValueError(f'Not one of the fs data file indicators: {name}')

    file = f'01 Carrington2007 FS data/data_{name}.dat'
    if not os.path.isfile(file):
        raise IOError(f'No fs data: File {file} missing.')

    with open(file) as f:
        for line in f:
            if 'x y' in line:
                break
        else:
            raise IOError(f'No labelline found in file {file}.')

        xx = []
        yy = []
        for line in f:
            if line:
                nums = [float(x) for x in line.split()]
                assert(len(nums) == 2)
                xx.append(nums[0])
                yy.append(nums[1])

    xx = np.array(xx) - 1
    yy = np.array(yy) - 1
    return xx * np.pi / A, yy * np.pi / B


def complete_fs_data(name):
    """ Get a completed branch of the fs, meaning a closed loop or full chain. """

    xx, yy = _import_fs_data_file(name)

    if name == 'chain':
        xx = np.array(list(xx) + list(-xx))
        yy = np.array(list(yy) * 2)
        order = np.argsort(xx)
        xx, yy = xx[order], yy[order]

    elif name in ['inner_square', 'outer_square', 'S']:
        xx = np.array(list(xx) + list(-xx)[::-1])
        yy = np.array(list(yy) + list(yy)[::-1])

        pphi = np.arccos(xx / np.sqrt(xx**2 + yy**2))
        pphi[yy < 0] *= -1
        order = np.argsort(pphi)
        xx, yy = xx[order], yy[order]

    elif name == 'iso_2d':
        pphi = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
        r = 1e9
        xx = r * np.cos(pphi)
        yy = r * np.sin(pphi)

    elif name == 'iso_1d':
        xx = np.linspace(-np.pi / A, np.pi / A, 1000, endpoint=False)
        yy = np.ones(len(xx)) * 3e9

    else:
        raise ValueError(f'No fs completion implemented for "{name}"')

    return xx, yy


def smooth(xx, yy, order, poly_order, modx):
    """ Can be a bit slow if you give a lot of data.

    Basically fit a polynomial on 'order' points nearby,
    must be odd. Then evaluate the polynomial at the
    central position and return the resulting y* value.

    A fast implementation would do the polynomial fits with
    a coefficient table or matrix, and update it point-to-point.
    But this is fast enough for a few hundred points and keeps it simple.

    ! Assumes xx to be increasing and modulo modx
    """

    N = len(xx)
    assert(len(yy) == N)
    assert(order % 2)
    assert(order > poly_order)
    assert(order < N)
    assert(all([late > prior for prior, late in zip(xx[:-1], xx[1:])]))
    assert(xx[-1] - xx[0] > modx * .8)

    new_y = np.zeros(N)
    for i in range(order // 2):
        xnow = list(xx[i - order // 2:] - modx) + list(xx[:i + order // 2 + 1])
        assert(len(xnow) == order)
        ynow = list(yy[i - order // 2:]) + list(yy[:i + order // 2 + 1])
        pp = np.polyfit(xnow, ynow, deg=poly_order)
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    for i in range(order // 2, N - order // 2):
        xnow = xx[i - order // 2:i + order // 2 + 1]
        assert(len(xnow) == order)
        ynow = yy[i - order // 2:i + order // 2 + 1]
        pp = np.polyfit(xnow, ynow, deg=poly_order)
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    for i in range(N - order // 2, N):
        xnow = list(xx[i - order // 2:] - modx) + \
            list(xx[:i - N + order // 2 + 1])
        assert(len(xnow) == order)
        ynow = list(yy[i - order // 2:]) + list(yy[:i - N + order // 2 + 1])
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    return new_y


def stabilised_deriv(xx, yy2, L, sm_length, modx):
    """ Derivative, smooth that, then integrate. Return a new yy
    with smoothed derivative in this way.

    yy2 is pre-smoothed such that interpolating -> derivative is ok.
    So perhaps a few points smoothing if the data is no good.

    L is the length to which yy2 is interpolated, think 10x your data.
    modx: see smooth() docs
    smooth is the order of the smoothing of the derivative.
        Think about the fraction of L [NOT len(xx)]
        Must be odd.
    """

    # Monotonically increasing x between 0 and modx
    assert(all(xj > xi for xi, xj in zip(xx[:-1], xx[1:])))
    assert(xx[0] >= 0)
    assert(xx[0] < modx / 10)
    assert(xx[-1] < modx)
    assert(xx[-1] > modx * 0.9)
    assert(sm_length % 2)
    assert(sm_length > 0)

    xfeed = list(xx - modx) + list(xx) + list(xx + modx)
    yfeed = list(yy2) * 3
    Interp = si.CubicSpline(xfeed, yfeed)
    x_int = np.linspace(0, modx, L * 2, endpoint=False)
    y_int = Interp(x_int)

    delta = modx * 1e-4
    dyy = (Interp(x_int + delta) - y_int) / delta

    stab_dyy = smooth(x_int, dyy, sm_length, 1, modx)
    stab_dyy += np.mean(dyy) - np.mean(stab_dyy)

    stabilised_xx = x_int[::2]
    stabilised_yy = np.zeros(len(stab_dyy) // 2)
    for i in range(1, len(stabilised_xx)):
        delta = stab_dyy[2 * i]
        delta += 2 * stab_dyy[(2 * i + 1) % len(stab_dyy)]
        delta += stab_dyy[(2 * i + 2) % len(stab_dyy)]
        delta *= 0.5 * (x_int[1] - x_int[0])
        stabilised_yy[i] = stabilised_yy[i - 1] + delta
    stabilised_yy += np.mean(y_int) - np.mean(stabilised_yy)

    return stabilised_xx, stabilised_yy


def make_stripped_cyclic_interpolation(xx, yy, modx):
    """ Cubic spline interpolation but instead of an object,
    get the breakpoints and coefficients of the polynomials.

    Repeats the data beyond both edges to ensure continuity
        of the value and first derivative also at the edge.
    """

    assert(len(xx) > 6)
    x_ext = list(xx[-6:-1] - modx) + list(xx) + list(xx[1:6] + modx)
    y_ext = list(yy[-6:-1]) + list(yy) + list(yy[1:6])

    Ppoly = si.CubicSpline(x_ext, y_ext)
    return Ppoly.x[5:-5], Ppoly.c[:, 5:-4]


@njit(i8(f8, f8[:]), cache=True)
def _binary_index_below(x, xxbreak):
    """ xxbreak[index] is the last one below x. Binary search O(ln(N)).

    Assumes xxbreak is monotonically increasing
    """

    assert(x >= xxbreak[0])

    # N is number of breakpoints
    # also N intervals (N-1 normal ones, and 1 for cyclic)
    N = len(xxbreak)
    maxi = N - 1
    mini = 0
    while maxi - mini > 1:
        middle = mini + (maxi - mini) // 2
        xmiddle = xxbreak[middle]
        if xmiddle - x > 0:
            maxi = middle
        else:
            mini = middle

    # Fix the last step of size 1 separately.
    index_below = maxi
    iters = 0
    while x < xxbreak[index_below]:
        index_below -= 1
        iters += 1
        assert(iters < 3)
    return index_below


@njit(f8(f8, f8[:], f8[:, :], f8), cache=True)
def eval_ppoly(x, xxbreak, cccoeff, modx):
    """ Evaluate the cubic spline interpolation at x.
    x may be given negative / above modx
    """

    x %= modx
    index_below = _binary_index_below(x, xxbreak)
    cc = cccoeff[:, index_below]
    order = len(cc) - 1
    y_eval = 0
    for i, c in enumerate(cc):
        y_eval += c * (x - xxbreak[index_below])**(order - i)
    return y_eval


@njit(f8(f8, f8[:], f8[:, :], f8), cache=True)
def eval_dppoly(x, xxbreak, cccoeff, modx):
    """ Evaluate the cubic spline interpolation derivative at x.
    x may be given negative / above modx
    """

    x %= modx
    index_below = _binary_index_below(x, xxbreak)
    cc = cccoeff[:, index_below]
    order = len(cc) - 1
    dy_eval = 0
    for i, c in enumerate(cc[:-1]):
        dy_eval += c * (x - xxbreak[index_below]
                        )**(order - i - 1) * (order - i)
    return dy_eval


def _re_interpolate_for_mirrors_2d(xxb, ccc):
    """ Only for 2d. Enforce tetragonal symmetry. """

    pphi = np.linspace(0, np.pi / 2, 500, endpoint=False)
    rr = [eval_ppoly(phi, xxb, ccc, 2 * np.pi) for phi in pphi]

    pphi2 = list(pphi) + list(np.pi - pphi)
    pphi2 += list(np.pi + pphi[1:]) + list(2 * np.pi - pphi)
    pphi2 = np.array(pphi2)
    rr2 = np.array(rr + rr + rr[1:] + rr)
    order = np.argsort(pphi2)
    pphi2 = pphi2[order]
    rr2 = rr2[order]

    xxb, ccc = make_stripped_cyclic_interpolation(pphi2, rr2, 2 * np.pi)
    return xxb, ccc


def smooth_fs_2d(name):
    """ Obtain a series of phi intervals with a 3rd degree polynomial
    defining the FS on each. Continuous value & 1st derivative. """

    if 'chain' in name or '1d' in name:
        raise ValueError('Only 2d Fermi surface sheets.')

    xx, yy = complete_fs_data(name)
    rr = np.sqrt(xx**2 + yy**2)
    pphi = np.arccos(xx / np.sqrt(xx**2 + yy**2))
    pphi[yy < 0] *= -1
    pphi %= 2 * np.pi
    order = np.argsort(pphi)
    rr = rr[order]
    pphi = pphi[order]

    # number of out-of-bound points to make sure smoothing through 0.
    # Choose at least half the initial-smooth order
    # (9//2 at the time of writing.)
    oob = 10
    pphi = list(pphi[-oob:] - 2 * np.pi) + \
        list(pphi) + list(pphi[:oob] + 2 * np.pi)
    pphi = np.array(pphi)
    rr = list(rr[-oob:]) + list(rr) + list(rr[:oob])
    rr = np.array(rr)

    rr2 = smooth(pphi, rr, 9, 1, 2 * np.pi)
    pphi3, rr3 = stabilised_deriv(pphi[oob:-oob], rr2[oob:-oob],
                                  1000, 31, 2 * np.pi)
    xxb, ccc = make_stripped_cyclic_interpolation(pphi3, rr3, 2 * np.pi)

    # This is very important to ensure sxy=0 at B=0.
    #   Without it, sxy can get a residual of sxx/2000 or so
    #   which is at 50 K equivalent to about 5 T in field.
    xxb, ccc = _re_interpolate_for_mirrors_2d(xxb, ccc)
    return xxb, ccc


def smooth_fs_1d(name):

    if 'square' in name or '2d' in name:
        raise ValueError('Only 1d Fermi surfaces')

    modx = 2 * np.pi / A
    xx, yy = complete_fs_data(name)

    xx %= modx
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]

    # Add a number of out-of-bound points to make sure smoothing through 0.
    # Choose at least half the initial-smooth order (9//2 at the time of writing.)
    oob = 10
    xxe = list(xx[-oob:] - modx) + list(xx) + list(xx[:oob] + modx)
    xxe = np.array(xxe)
    yye = list(yy[-oob:]) + list(yy) + list(yy[:oob])
    yye = np.array(yye)

    yy2 = smooth(xxe, yye, 9, 1, modx)
    xx3, yy3 = stabilised_deriv(xxe[oob:-oob], yy2[oob:-oob], 1000, 31, modx)
    xxb, ccc = make_stripped_cyclic_interpolation(xx3, yy3, modx)
    return xxb, ccc


#############################
# Initialise fs
# FS and mean free path
#############################


@njit(f8(f8, f8[:], f8[:, :]), cache=True)
def get_r_2d(phi, xxb, ccc):
    """ Point in k-space on the inner substantial square 2d fs """
    return eval_ppoly(phi, xxb, ccc, 2 * np.pi)


@njit(f8[:](f8, f8[:], f8[:, :]), cache=True)
def get_Ldir_2d(phi, xxb, ccc):
    """ Get the 2d normalised direction of the L vector at this fs point. """
    deriv = eval_dppoly(phi, xxb, ccc, 2 * np.pi)
    r = get_r_2d(phi, xxb, ccc)
    Ldir = np.array([np.cos(phi), np.sin(phi)])
    Ldir -= np.array([-np.sin(phi), np.cos(phi)]) * deriv / r
    return Ldir / np.linalg.norm(Ldir)


@njit(f8(f8, f8[:], f8[:], f8[:, :]), cache=True)
def get_L_2d(phi, Lset, xxb, ccc):
    """ Get the *size* of the mean free path. """

    if Lset[0] == 0:
        # Isotropic L
        return Lset[1]

    elif Lset[0] == 1:
        # Initial artificial anisotropy
        r = get_r_2d(phi, xxb, ccc)
        return (r - Lset[2]) / 1e9 * Lset[1]

    elif Lset[0] == 2:
        # Impeded orbital motion
        dphi = phi % (2 * np.pi)
        dphi = min(dphi, 2 * np.pi - dphi)
        e1 = np.exp(-dphi**2 / 2 / 0.03**2)
        dphi = (phi - np.pi / 2) % (2 * np.pi)
        dphi = min(dphi, 2 * np.pi - dphi)
        e2 = np.exp(-dphi**2 / 2 / 0.03**2)
        dphi = (phi - np.pi) % (2 * np.pi)
        dphi = min(dphi, 2 * np.pi - dphi)
        e3 = np.exp(-dphi**2 / 2 / 0.03**2)
        dphi = (phi - 3 * np.pi / 2) % (2 * np.pi)
        dphi = min(dphi, 2 * np.pi - dphi)
        e4 = np.exp(-dphi**2 / 2 / 0.03**2)
        g_cold = 1 / Lset[1]
        g_hot = 1 / Lset[2] * (e1 + e2 + e3 + e4)
        return 1 / (g_cold + g_hot)

    elif Lset[0] == 3:
        # Lset has [3, Lchain, Lplane] both isotropic
        # with a small linear-in-phi transition between.
        # Specifically made for outer_square in YBCO.
        # Map to first quadrant
        phi = phi % np.pi
        if phi > np.pi / 2:
            phi = np.pi - phi
        if phi < 57 / 180 * np.pi:
            return Lset[2]
        elif phi < 63 / 180 * np.pi:
            deg = phi * 180 / np.pi - 57
            return Lset[2] + deg / 6 * (Lset[1] - Lset[2])
        else:
            return Lset[1]
    assert(False)


@njit(f8[:](f8, f8[:], f8[:], f8[:, :]), cache=True)
def get_Lvec_2d(phi, Lset, xxb, ccc):
    """ Convenience. The full vector. """
    Ldir = get_Ldir_2d(phi, xxb, ccc)
    Lval = get_L_2d(phi, Lset, xxb, ccc)
    return Lval * Ldir


@njit(f8(f8, f8[:], f8[:, :]), cache=True)
def get_ky_1d(kx, xxb, ccc):
    """ Point in k-space on the chain 1d fs """
    return eval_ppoly(kx, xxb, ccc, 2 * np.pi / A)


@njit(f8[:](f8, f8[:], f8[:, :]), cache=True)
def get_Ldir_1d(kx, xxb, ccc):
    """ Get the 2d normalised direction of the L vector at this fs point. """
    deriv = eval_dppoly(kx, xxb, ccc, 2 * np.pi / A)
    if deriv == 0:
        return np.array([0, 1.])
    Ldir = np.array([1, -1 / deriv])
    Ldir *= -1 + 2 * (Ldir[1] > 0)
    return -Ldir / np.linalg.norm(Ldir)


@njit(f8(f8, f8[:], f8[:], f8[:, :]), cache=True)
def get_L_1d(kx, Lset, xxb, ccc):
    """ Get the *size* of the mean free path. """

    if Lset[0] == 0:
        # Isotropic L
        return Lset[1]
    elif Lset[0] == 1:
        # Some anisotropy at the start
        ky = get_ky_1d(kx, xxb, ccc)
        return (abs(ky) - Lset[2]) / 2e9 * Lset[1]
    # No impeded orbital motion in 1d implemented.
    elif Lset[0] == 3:
        # Lchain and Lpocket at different parts.
        # Lset[1] is chain, Lset[2] is pocket
        # Specific to the YBCO chain.
        kx = ((kx + np.pi / A) % (2 * np.pi / A)) - np.pi / A
        if kx < -4.2e9:
            return Lset[1]
        elif kx < -3.5e9:
            return Lset[1] + (kx + 4.2e9) / 7e8 * (Lset[2] - Lset[1])
        elif kx < 3.5e9:
            return Lset[2]
        elif kx < 4.2e9:
            return Lset[1] + (4.2e9 - kx) / 7e8 * (Lset[2] - Lset[1])
        else:
            return Lset[1]
    assert(False)


@njit(f8[:](f8, f8[:], f8[:], f8[:, :]), cache=True)
def get_Lvec_1d(phi, Lset, xxb, ccc):
    """ Convenience. The full vector. """
    Ldir = get_Ldir_1d(phi, xxb, ccc)
    Lval = get_L_1d(phi, Lset, xxb, ccc)
    return Lval * Ldir


@njit(f8(f8[:], f8[:, :]), cache=True)
def compute_density_2d(xxb, ccc):
    """ Get the density in 1/m^3 assuming spin degeneracy """

    # Just use 10k points.
    # That is about 1e-8 relative error, nobody cares at this level, it is still fast.
    k_area = 0
    pphi = np.linspace(0, 2 * np.pi, 10000)
    rr = [get_r_2d(phi, xxb, ccc) for phi in pphi]
    for i in range(len(pphi) - 1):
        k_area += 0.5 * (rr[i] + rr[i + 1])**2 / 4 * (pphi[i + 1] - pphi[i])
    # 2 for spin
    # 8pi^3 Fourier
    # 2pi/C*area is volume in the FS
    n = 2 / (8 * np.pi**3) * 2 * np.pi / C * k_area
    return n


def auto_density_2d(name):
    """ Get the density in 1/m^3 assuming spin degeneracy """
    xxb, ccc = smooth_fs_2d(name)
    return compute_density_2d(xxb, ccc)


def show_smooth_fs_1d(name, xxb, ccc, Lset, *, size_factor='auto', ax='new'):
    """ Show the Fermi surface and mean free path configured in globals.

    Length of mean free path is L * size_factor. About 1e17 ish.
    """

    xx, yy = complete_fs_data(name)
    xplt = np.linspace(-np.pi / A, np.pi / A, 10000)
    yplt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi / A) for x in xplt])

    if ax == 'new':
        f, ax = plt.subplots(num=f'fs {name}')
        ax.scatter(xx, yy, color='tab:blue')

    ax.plot(xplt, yplt, color='tab:blue', lw=4, zorder=2)
    ax.plot(-xplt, -yplt, color='tab:blue', lw=4, zorder=2)
    ax.set_aspect('equal')

    if size_factor == 'auto':
        size_factor = 1e17 * 1e-8 / min(Lset[1:])

    for i in range(0, len(xplt), len(xplt) // 150):
        if xplt[i] >= -np.pi / A and xplt[i] <= np.pi / A:
            Ldir = get_Ldir_1d(xplt[i], xxb, ccc)
            L = Ldir * get_L_1d(xplt[i], Lset, xxb, ccc)
            ax.plot([xplt[i], xplt[i] + L[0] * size_factor],
                    [yplt[i], yplt[i] + L[1] * size_factor], 
                    color='tab:blue', lw=1, zorder=1)
            ax.plot([-xplt[i], -xplt[i] - L[0] * size_factor],
                    [-yplt[i], -yplt[i] - L[1] * size_factor], 
                    color='tab:blue', lw=1, zorder=1)


def show_smooth_Lorb_1d(name, xxb, ccc, Lset):
    """ See Ong1991 for relevance to Hall effect. """

    xplt = np.linspace(-np.pi / A, np.pi / A, 10000)
    yplt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi / A) for x in xplt])

    LL = []
    for i in range(0, len(xplt)):
        if xplt[i] >= -np.pi / A and xplt[i] <= np.pi / A:
            Ldir = get_Ldir_1d(xplt[i], xxb, ccc)
            L = Ldir * get_L_1d(xplt[i], Lset, xxb, ccc)
            LL.append(L)

    f, ax = plt.subplots(figsize=(10, 8))
    Lxx, Lyy = np.array(LL).T
    f.suptitle(f'Fermi surface {name}')
    ax.scatter(Lxx * 1e9, Lyy * 1e9, lw=4, color='tab:blue')
    ax.set_xlabel('$l_x$ (nm)')
    ax.set_ylabel('$l_y$ (nm)')
    ax.set_aspect('equal')
    return f, ax


def show_smooth_fs_2d(name, xxb, ccc, Lset, *, size_factor='auto', ax='new'):
    """ Visualise the digitized and smoothed FS and Ldir across. """

    phi_plt = np.linspace(-1, 2 * np.pi + 1, 10000)
    r_plt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi) for x in phi_plt])
    dplt = [eval_dppoly(phi, xxb, ccc, 2 * np.pi) for phi in phi_plt]

    xx, yy = complete_fs_data(name)
    if ax == 'new':
        f, ax = plt.subplots(num=f'fs {name}')
        ax.scatter(xx, yy, color='tab:blue')

    color = 'tab:red' if name == 'inner_square' else 'tab:green'
    ax.plot(r_plt * np.cos(phi_plt), 
            r_plt * np.sin(phi_plt), color=color, lw=4, zorder=2)
    ax.set_aspect('equal')

    if size_factor == 'auto':
        size_factor = 1e17 * 1e-8 / min(Lset[1:])

    for i in range(0, len(dplt), len(dplt) // 300):
        phi = phi_plt[i]
        if phi >= 0 and phi < 2 * np.pi:
            Ldir = get_Ldir_2d(phi, xxb, ccc)
            L = Ldir * get_L_2d(phi, Lset, xxb, ccc)
            pp = np.array([r_plt[i] * np.cos(phi), r_plt[i] * np.sin(phi)])
            ax.plot([pp[0], pp[0] + L[0] * size_factor],
                    [pp[1], pp[1] + L[1] * size_factor], 
                    color=color, zorder=1, lw=1)


def show_smooth_Lorb_2d(name, xxb, ccc, Lset):
    """ See Ong1991 for relevance to Hall effect. """

    phi_plt = np.linspace(-1, 2 * np.pi + 1, 10000)
    r_plt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi) for x in phi_plt])
    dplt = [eval_dppoly(phi, xxb, ccc, 2 * np.pi) for phi in phi_plt]

    color = 'tab:red' if name == 'inner_square' else 'tab:green'

    LL = []
    for i in range(0, len(dplt)):
        phi = phi_plt[i]
        if phi >= 0 and phi < 2 * np.pi:
            Ldir = get_Ldir_2d(phi, xxb, ccc)
            L = Ldir * get_L_2d(phi, Lset, xxb, ccc)
            LL.append(L)

    f, ax = plt.subplots(figsize=(10, 8))
    Lxx, Lyy = np.array(LL).T
    f.suptitle(f'L orbit for {name}')
    ax.scatter(Lxx * 1e9, Lyy * 1e9, lw=4, color='tab:blue')
    ax.set_xlabel('$l_x$ (nm)')
    ax.set_ylabel('$l_y$ (nm)')
    ax.set_aspect('equal')
    return f, ax



#############################
# FS and mean free path
# Pathlength 2d
#############################


@njit(f8(f8, f8, f8, f8, f8[:], f8[:, :], f8))
def _rec_pathlength_euc_2d(phi1, r1, phi2, r2, xxb, ccc, err):
    """ Private. """

    dist = np.sqrt((r1 * np.cos(phi1) - r2 * np.cos(phi2))**2 +
                   (r1 * np.sin(phi1) - r2 * np.sin(phi2))**2)

    half = .5 * (phi1 + phi2)
    rh = get_r_2d(half, xxb, ccc)
    dist2 = np.sqrt((r1 * np.cos(phi1) - rh * np.cos(half))**2
                    + (r1 * np.sin(phi1) - rh * np.sin(half))**2)
    dist2 += np.sqrt((r2 * np.cos(phi2) - rh * np.cos(half))**2
                     + (r2 * np.sin(phi2) - rh * np.sin(half))**2)

    if abs(dist - dist2) > err * dist2:
        dA = _rec_pathlength_euc_2d(phi1, r1, half, rh, xxb, ccc, err)
        dB = _rec_pathlength_euc_2d(half, rh, phi2, r2, xxb, ccc, err)
        return dA + dB
    return dist2


@njit(f8(f8, f8, f8[:], f8[:, :], f8))
def get_pathlength_euc_2d(phi1, phi2, xxb, ccc, err):
    """ Get the pathlength to err relative tolerance.

    Adds points and uses straight-distance.
    """

    r1 = get_r_2d(phi1, xxb, ccc)
    r2 = get_r_2d(phi2, xxb, ccc)
    return _rec_pathlength_euc_2d(phi1, r1, phi2, r2, xxb, ccc, err)


@njit(f8[:](f8, f8, f8, f8, f8, f8))
def fit_parabola(x1, x2, x3, y1, y2, y3):
    """ Get vec(a):  a0+a1*x+a2*x**2 = y fit through these 3 points """

    # y = A alpha
    # where alpha are the 3 coefficients to fit
    # Perhaps sped up if written explicitly instead of using linalg.inv()?
    A = np.array([[1, x1, x1**2],
                  [1, x2, x2**2],
                  [1, x3, x3**2]])
    return np.linalg.inv(A) @ np.array([y1, y2, y3])


@njit(f8(f8, f8, f8[:]))
def measure_pathlength_parabola(x1, x3, alphas):
    """ Pathlength of the fit_parabola result.

    Assumes x's are the same as the fit and extrema.
    """

    term = alphas[1] + alphas[2] * x3
    v1 = (np.sqrt(term**2 + 1) * term + np.arcsinh(term)) / (2 * alphas[2])
    term = alphas[1] + alphas[2] * x1
    v2 = (np.sqrt(term**2 + 1) * term + np.arcsinh(term)) / (2 * alphas[2])
    return abs(v1 - v2)


@njit(f8(f8, f8, f8, f8, f8, f8))
def fit_and_measure_pathlength(x1, x2, x3, y1, y2, y3):
    """ Calculate the pathlength of these points by drawing
    a parabola through them.

    Automatically applies a basis transformation in case of
    retrograde x. Points are visited 1->2->3 (or vice versa)
    """

    x2 -= x1
    x3 -= x1
    x1 = 0
    y3 -= y1
    y2 -= y1
    y1 = 0

    b1 = np.array([x3 - x1, y3 - y1])
    b1 /= np.linalg.norm(b1)
    b2 = np.array([b1[1], -b1[0]])

    px = b1[0] * x1 + b1[1] * y1
    py = b2[0] * x1 + b2[1] * y1

    ux = b1[0] * x2 + b1[1] * y2
    uy = b2[0] * x2 + b2[1] * y2

    vx = b1[0] * x3 + b1[1] * y3
    vy = b2[0] * x3 + b2[1] * y3

    assert(vx > ux)
    assert(ux > px)
    alphas = fit_parabola(px, ux, vx, py, uy, vy)
    return measure_pathlength_parabola(px, vx, alphas)


@njit(f8(f8, f8, f8, f8, f8, f8, f8[:], f8[:, :], f8))
def _rec_pathlength_2d(phi1, r1, phi2, r2, phi3, r3, xxb, ccc, err):
    """ Private """

    x1 = r1 * np.cos(phi1)
    y1 = r1 * np.sin(phi1)
    x2 = r2 * np.cos(phi2)
    y2 = r2 * np.sin(phi2)
    x3 = r3 * np.cos(phi3)
    y3 = r3 * np.sin(phi3)
    dist = fit_and_measure_pathlength(x1, x2, x3, y1, y2, y3)

    half12 = .5 * (phi1 + phi2)
    r12 = get_r_2d(half12, xxb, ccc)
    x12 = r12 * np.cos(half12)
    y12 = r12 * np.sin(half12)
    distA = fit_and_measure_pathlength(x1, x12, x2, y1, y12, y2)

    half23 = .5 * (phi2 + phi3)
    r23 = get_r_2d(half23, xxb, ccc)
    x23 = r23 * np.cos(half23)
    y23 = r23 * np.sin(half23)
    distB = fit_and_measure_pathlength(x2, x23, x3, y2, y23, y3)

    if abs(dist - distA - distB) > err * dist:
        dA = _rec_pathlength_2d(
            phi1, r1, half12, r12, phi2, r2, xxb, ccc, err)
        dB = _rec_pathlength_2d(
            phi2, r2, half23, r23, phi3, r3, xxb, ccc, err)
        return dA + dB
    return distA + distB


@njit(f8(f8, f8, f8[:], f8[:, :], f8))
def get_pathlength_par_2d(phi1, phi2, xxb, ccc, err):
    """ Get the pathlength between these endpoints.
    Fits segments with parabolas for better complexity than straight lines.
    """

    assert(phi2 - phi1 < 2)
    assert(phi2 - phi1 > -2)

    r1 = get_r_2d(phi1, xxb, ccc)
    half = .5 * (phi1 + phi2)
    r12 = get_r_2d(half, xxb, ccc)
    r2 = get_r_2d(phi2, xxb, ccc)
    r = _rec_pathlength_2d(phi1, r1, half, r12, phi2, r2, xxb, ccc, err)
    return r


def test_pathlength_performance():
    """ Compare the Euclidean and parabolic convergence. """


# Better performance despite slower convergence.
#   Parabolic is better below 1e-10 error.
#   The fs points are just very cheap here.
get_pathlength_2d = get_pathlength_euc_2d


#############################
# Pathlength 2d
# Survival only 2d
#############################

@njit(f8(f8, f8, f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def _survive_2d(phi1, L1, phi2, L2, B, Lset, xxb, ccc, err):
    """ Private. Euclidean. """

    half = 0.5 * (phi1 + phi2)
    Lh = get_L_2d(half, Lset, xxb, ccc)
    klength1 = get_pathlength_2d(phi1, half, xxb, ccc, err)
    klength2 = get_pathlength_2d(half, phi2, xxb, ccc, err)

    integral = (1 / L1 + 1 / L2) * (klength1 + klength2) / 2
    whole = np.exp(-HBAR / (Q * B) * integral)

    integral = (1 / L1 + 1 / Lh) * klength1 / 2
    bisected = np.exp(-HBAR / (Q * B) * integral)
    integral = (1 / Lh + 1 / L2) * klength2 / 2
    bisected *= np.exp(-HBAR / (Q * abs(B)) * integral)

    if bisected < 1e-10:
        # Who cares if it is 1e-10 or 1e-100.
        return 0
    if abs(bisected - whole) < err and abs(bisected - whole) < 0.1 * bisected:
        return bisected
    P1 = _survive_2d(phi1, L1, half, Lh, B, Lset, xxb, ccc, err)
    P2 = _survive_2d(half, Lh, phi2, L2, B, Lset, xxb, ccc, err)
    return P1 * P2


@njit(f8(f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def survive_2d(phi1, phi2, B, Lset, xxb, ccc, err):
    """ Compute the odds to traverse between these phis.

    Does NOT care about direction of cyclotron motion.
    Evaluates the following integral:
        P = exp(-hbar/qB * integral_dkphi 1/L )

    Because magnetic field is along c and the mean free path
    is in the xy plane, no inner products are required.
    """

    assert(phi1 - phi2 < 2)
    assert(phi1 - phi2 > -2)

    L1 = get_L_2d(phi1, Lset, xxb, ccc)
    L2 = get_L_2d(phi2, Lset, xxb, ccc)
    return _survive_2d(phi1, L1, phi2, L2, B, Lset, xxb, ccc, err)


def show_survival_2d(name, B, Lset, *, show=True):
    """ Show the survival odds. """

    xxb, ccc = smooth_fs_2d(name)

    pphi = np.linspace(0, 2 * np.pi, 1000)
    pprob = np.zeros(len(pphi) - 1)
    for i in range(len(pphi) - 1):
        pprob[i] = survive_2d(pphi[i], pphi[i + 1], B, Lset, xxb, ccc, 1e-5)

    survival = np.ones(len(pphi))
    for i, prob in enumerate(pprob):
        survival[i + 1] = survival[i] * prob

    if show:
        show_smooth_fs_2d(name, xxb, ccc, Lset)

        plt.figure('Survival starting at phi=0 ccw')
        plt.plot(pphi, survival)
        plt.xlabel('$\u03D5$ (rad)')
        plt.ylabel('$P$')
        plt.xlim(0, 2 * np.pi)
        plt.ylim(0, 1)

        plt.figure('wctau')
        plt.plot(.5 * (pphi[1:] + pphi[:-1]),
                 pprob ** (1 / (pphi[1:] - pphi[:-1])))
        plt.xlabel('$\u03D5$ (rad)')
        plt.ylabel('$\u03C9_c\u03C4$ (rad)')
        plt.xlim(0, 2 * np.pi)

    loop = 1
    for prob in pprob:
        loop *= prob
    return loop


#############################
# Survival only 2d
# Cyclotron integral 2d
#############################


@njit(f8[:](f8, f8[:], f8, f8[:], f8, f8, f8, f8[:], f8[:], f8[:, :]))
def _survive_step_plus_Lpartial(phi1, Lvec1, phi2, Lvec2, kpath, odds, B, Lset, xxb, ccc):
    """ Private.

    Return the probability to survive this step as well as
    the real space distance (weighed by survival).
    Single step, so this is only valid if the resulting probability
    is substantial!

    Euler integration because I am too lazy for this code that will
    be fast enough to implement a function that finds the exact
    halfway point between phi1 and phi2. I mean halfway in k distance
    *along the surface* efficiently.

    kpath = get_pathlength_2d(phi1, phi2, xxb, ccc, err)
    is presumed to have integrity.
    This pre-computation prevents this being re-computed every time.

    odds is the change to even start this interval.
    """

    iLavg = 0.5 * (1 / np.linalg.norm(Lvec1) + 1 / np.linalg.norm(Lvec2))
    prob = np.exp(-HBAR / (abs(B) * Q) * kpath * iLavg)

    # Integral is usually written P(0) * integral(vvec * P(t) dt)
    #   where P(0)=odds are the odds to reach this sector.
    #   Here expressed in purely mean free paths by transforming t -> k basis.
    #   This is physically the contribution to the mean free path including
    #   the impact of the magnetic field.
    integral = odds * Lvec1 / np.linalg.norm(Lvec1)
    integral += odds * prob * Lvec2 / np.linalg.norm(Lvec2)
    integral *= HBAR * kpath / (Q * abs(B)) / 2
    return np.array([prob, integral[0], integral[1]])


@njit(f8[:](f8, f8[:], f8, f8[:], f8, f8[:], f8, f8, f8, f8, f8[:], f8[:], f8[:, :]))
def _survive_step_plus_Lpartial3(phi1, Lvec1, phi2, Lvec2, phi3, Lvec3,
                                 kpath12, kpath23, odds, B, Lset, xxb, ccc):
    """ Private. Simpson version  """

    iLavg = 0.5 * (1 / np.linalg.norm(Lvec1) + 1 / np.linalg.norm(Lvec2))
    prob12 = np.exp(-HBAR / (abs(B) * Q) * kpath12 * iLavg)
    iLavg = 0.5 * (1 / np.linalg.norm(Lvec2) + 1 / np.linalg.norm(Lvec3))
    prob23 = np.exp(-HBAR / (abs(B) * Q) * kpath23 * iLavg)

    # Integral is usually written P(0) * integral(vvec * P(t) dt)
    #   where P(0)=odds are the odds to reach this sector.
    #   Here expressed in purely mean free paths by transforming t -> k basis.
    #   This is physically the contribution to the mean free path including
    #   the impact of the magnetic field.
    # Fit y = a*k^2 + b*k + c on the integrand (here named y for ease)
    # and integrate this fit to get Simpson on unequal spacing.
    assert(kpath12 > 0)
    assert(kpath23 > 0)
    a, b, c = fit_parabola(0, kpath12, kpath12 + kpath23,
                           Lvec1[0] / np.linalg.norm(Lvec1),
                           Lvec2[0] / np.linalg.norm(Lvec2) * prob12,
                           Lvec3[0] / np.linalg.norm(Lvec3) * prob12 * prob23)
    kp = kpath12 + kpath23
    integral_x = a * kp + .5 * b * kp**2 + 1 / 3 * c * kp**2
    integral_x *= odds * HBAR / (Q * abs(B))

    a, b, c = fit_parabola(0, kpath12, kpath12 + kpath23,
                           Lvec1[1] / np.linalg.norm(Lvec1),
                           Lvec2[1] / np.linalg.norm(Lvec2) * prob12,
                           Lvec3[1] / np.linalg.norm(Lvec3) * prob12 * prob23)
    kp = kpath12 + kpath23
    integral_y = a * kp + .5 * b * kp**2 + 1 / 3 * c * kp**2
    integral_y *= odds * HBAR / (Q * abs(B))

    return np.array([prob12 * prob23, integral_x, integral_y])


@njit(f8[:](f8, f8, f8, f8[:], f8[:], f8[:, :], f8, f8, f8, f8))
def _rec_partial_Lvec(phi1, phi2, B, Lset, xxb, ccc, err, odds, int_x, int_y):
    """ Private """

    assert(abs(phi1 - phi2) > 1e-20)
    if odds < err / 10:
        return np.array([0., 0, 0])

    # One angle to significantly speed up the program:
    #   Remove the nested recursion.
    #   Use a fixed number of points for get_pathlength_2d
    #   and compute k12 separately instead of k1h+kh2.
    #   Then use just 1 optimisation loop.
    # The reason it is not done here is because this code is
    #   testable, nice and fast enough. If you do this
    #   in 3d or with angled field then you might worry.
    half = .5 * (phi1 + phi2)
    k1h = get_pathlength_2d(phi1, half, xxb, ccc, err)
    kh2 = get_pathlength_2d(half, phi2, xxb, ccc, err)
    Lv1 = get_Lvec_2d(phi1, Lset, xxb, ccc)
    Lvh = get_Lvec_2d(half, Lset, xxb, ccc)
    Lv2 = get_Lvec_2d(phi2, Lset, xxb, ccc)

    # Standard technique: Compare Euler to Simpson for error estimate.
    # Use the Simpson result.
    prob, ix, iy = _survive_step_plus_Lpartial(phi1, Lv1, phi2, Lv2, k1h + kh2,
                                               odds, B, Lset, xxb, ccc)
    prob2, ix2, iy2 = _survive_step_plus_Lpartial3(phi1, Lv1, half, Lvh,
                                                   phi2, Lv2, k1h, kh2, odds,
                                                   B, Lset, xxb, ccc)

    # Only track the relative error in the *leading* component.
    #   One may be zero for all I know.
    if abs(int_x + ix2) > abs(int_y + iy2):
        diff = abs(ix2 - ix) / abs(int_x + ix2) * 100
    else:
        diff = abs(iy2 - iy) / abs(int_y + iy2) * 100

    # The diff condition basically tests if this present interval
    #   violates the *total* err on all intervals combined
    #   given the value of the rolling integrals.
    #   Add the factor 100 just to make this more towards what the
    #   *cumulative* error would be.
    # The probability is a rolling error and so this part has to be
    #   strict particularly in early intervals. Measure this
    #   by the error made in the absolute change in survival odds.
    # However, printing its data it seems to always be satisfied?
    # print(diff, prob, prob2, '|', np.linalg.norm(Lv1), np.linalg.norm(Lvh), np.linalg.norm(Lv2))
    if diff < err and odds * abs(prob - prob2) < err / 100 and prob > 0.1:
        return np.array([prob2, ix2, iy2])
    else:
        probC, ixC, iyC = _rec_partial_Lvec(phi1, half, B, Lset, xxb, ccc,
                                            err, odds, int_x, int_y)
        probD, ixD, iyD = _rec_partial_Lvec(half, phi2, B, Lset, xxb, ccc,
                                            err, odds * probC,
                                            int_x + ixC, int_y + iyC)
        return np.array([probC * probD, ixC + ixD, iyC + iyD])


@njit(f8[:](f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def partial_Lvec_2d(phi1, phi2, B, Lset, xxb, ccc, err):
    """ Compute the distance a quasi-particle traverses until the next interval
    on the Fermi surface, as well as the probability to reach that point. """

    assert(abs(phi1 - phi2) < 2)
    assert(phi2 > phi1)
    return _rec_partial_Lvec(phi1, phi2, B, Lset, xxb, ccc, err, 1, 0, 0)


@njit(f8[:, :](f8[:], f8, f8[:], f8[:], f8[:, :], f8))
def partial_Lvecs_2d(pphi, B, Lset, xxb, ccc, err):
    """ Convenience.
    Specifically for the full pocket, i.e. pphi starts at 0 and ends at 2pi.
    """

    assert(pphi[0] == 0)
    assert(abs(pphi[-1] - 2 * np.pi) < 1e-12)

    N = len(pphi)
    odds = np.zeros(N - 1)
    pLxx = np.zeros(N - 1)
    pLyy = np.zeros(N - 1)
    for i in range(N - 1):
        odds[i], pLxx[i], pLyy[i] = partial_Lvec_2d(pphi[i], pphi[i + 1],
                                                    B, Lset, xxb, ccc, err)

    r = np.empty((3, N - 1))
    r[0] = odds
    assert(max(odds) < 1)
    assert(min(odds) >= 0)
    r[1] = pLxx
    r[2] = pLyy
    return r


@njit(f8[:, :](f8[:], f8[:], f8[:]))
def convolve_partial_Lvecs(odds, pLxs, pLys):
    """ Given the odds to live and partial Lx/Ly for each phi-interval,
    compute the full Lxs and Lys. L here refers to integral_dt(v(t)P(t))
    where v is velocity and P probability to live, including the influence
    of the magnetic field. So not the mean free path, but the mean free
    path reduced by the curved motion under the magnetic field,
    a sort of net displacement in the real crystal.

    ! Assumes the intervals combined span the full Fermi surface.
    So prod(odds) is really the probability to survive a full orbit.

    """

    N = len(pLxs)
    Lxx = np.zeros(N)
    Lyy = np.zeros(N)

    # First traverse the first one step by step,
    # moving interval by interval till odds have deteriorated enough.
    cutoff = 1e-10
    survival = 1
    i = 0
    iteration = 0
    while survival > cutoff:
        Lxx[0] += pLxs[i] * survival
        Lyy[0] += pLys[i] * survival
        survival *= odds[i]
        i = (i + 1) % N
        iteration += 1
        assert(iteration < N * 1000)

    # Next, work backwords and use this one full result
    # to quickly get the rest.
    j = N - 1
    while j > 0:
        Lxx[j] = pLxs[j] + odds[j] * Lxx[(j + 1) % N]
        Lyy[j] = pLys[j] + odds[j] * Lyy[(j + 1) % N]
        j -= 1

    result = np.zeros((2, N))
    result[0] = Lxx
    result[1] = Lyy
    return result


#############################
# Cyclotron integral 2d
# Pathlength and survival 1d
#############################


@njit(f8(f8, f8, f8, f8, f8[:], f8[:, :], f8))
def _rec_pathlength_euc_1d(kx1, ky1, kx2, ky2, xxb, ccc, err):
    """ Private. """

    assert(abs(kx1 - ky1) > 1e-5)

    dist = np.sqrt((kx1 - kx2)**2 + (ky1 - ky2)**2)
    kxh = .5 * (kx1 + kx2)
    kyh = get_ky_1d(kxh, xxb, ccc)

    dist2 = np.sqrt((kx1 - kxh)**2 + (ky1 - kyh)**2)
    dist2 += np.sqrt((kxh - kx2)**2 + (kyh - ky2)**2)
    if abs(dist - dist2) > err * dist2:
        dA = _rec_pathlength_euc_1d(kx1, ky1, kxh, kyh, xxb, ccc, err)
        dB = _rec_pathlength_euc_1d(kxh, kyh, kx2, ky2, xxb, ccc, err)
        return dA + dB
    return dist2


@njit(f8(f8, f8, f8[:], f8[:, :], f8))
def get_pathlength_euc_1d(kx1, kx2, xxb, ccc, err):
    """ along the curved path between kx1 and kx2.
    No modulos, directly kx1 and kx2. """
    ky1 = get_ky_1d(kx1, xxb, ccc)
    ky2 = get_ky_1d(kx2, xxb, ccc)
    return _rec_pathlength_euc_1d(kx1, ky1, kx2, ky2, xxb, ccc, err)


# Given that euclidean was fine for 2d in overall speed to get sigma,
# and the requirement to make that matrix inverse more efficient
# to get parabolic to even compete for ~1e-6 accuracy leaves me to just
# implement euclidean here.
get_pathlength_1d = get_pathlength_euc_1d


@njit(f8(f8, f8, f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def _survive_1d(kx1, L1, kx2, L2, B, Lset, xxb, ccc, err):
    """ Private. Euclidean. """

    assert(abs(kx1 - kx2) > 1)
    half = 0.5 * (kx1 + kx2)
    Lh = get_L_1d(half, Lset, xxb, ccc)
    klength1 = get_pathlength_1d(kx1, half, xxb, ccc, err)
    klength2 = get_pathlength_1d(half, kx2, xxb, ccc, err)

    integral = (1 / L1 + 1 / L2) * (klength1 + klength2) / 2
    whole = np.exp(-HBAR / (Q * B) * integral)

    integral = (1 / L1 + 1 / Lh) * klength1 / 2
    bisected = np.exp(-HBAR / (Q * B) * integral)
    integral = (1 / Lh + 1 / L2) * klength2 / 2
    bisected *= np.exp(-HBAR / (Q * abs(B)) * integral)

    if bisected < 1e-10:
        # Who cares if it is 1e-10 or 1e-100.
        return 0
    if abs(bisected - whole) < err and abs(bisected - whole) < 0.1 * bisected:
        return bisected
    P1 = _survive_1d(kx1, L1, half, Lh, B, Lset, xxb, ccc, err)
    P2 = _survive_1d(half, Lh, kx2, L2, B, Lset, xxb, ccc, err)
    return P1 * P2


@njit(f8(f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def survive_1d(kx1, kx2, B, Lset, xxb, ccc, err):
    """ Compute the odds to traverse between these phis.

    Does NOT care about direction of cyclotron motion.
    Evaluates the following integral:
        P = exp(-hbar/qB * integral_dkphi 1/L )

    Because magnetic field is along c and the mean free path
    is in the xy plane, no inner products are required.
    """

    L1 = get_L_1d(kx1, Lset, xxb, ccc)
    L2 = get_L_1d(kx2, Lset, xxb, ccc)
    return _survive_1d(kx1, L1, kx2, L2, B, Lset, xxb, ccc, err)


def show_survival_1d(name, B, Lset, *, show=True):
    """ Show the survival odds. """

    xxb, ccc = smooth_fs_1d(name)

    kkx = np.linspace(0, 2 * np.pi / A, 1000)
    pprob = np.zeros(len(kkx) - 1)
    for i in range(len(kkx) - 1):
        pprob[i] = survive_1d(kkx[i], kkx[i + 1], B, Lset, xxb, ccc, 1e-5)

    survival = np.ones(len(kkx))
    for i, prob in enumerate(pprob):
        survival[i + 1] = survival[i] * prob

    if show:
        show_smooth_fs_1d(name, xxb, ccc, Lset)

        plt.figure('Survival starting at phi=0 ccw')
        plt.plot(kkx, survival)
        plt.xlabel('$\u03D5$ (rad)')
        plt.ylabel('$P$')
        plt.xlim(0, 2 * np.pi / A)
        plt.ylim(0, 1)

        plt.figure('wctau')
        plt.plot(.5 * (kkx[1:] + kkx[:-1]), pprob **
                 (1 / A / (kkx[1:] - kkx[:-1])))
        plt.xlabel('$kx$ (1/m)')
        plt.ylabel('$\u03C9_c\u03C4_{eff}$ (rad)')
        plt.xlim(0, 2 * np.pi / A)

    loop = 1
    for prob in pprob:
        loop *= prob
    return loop


#############################
# Pathlength and survival 1d
# Displacement 1d
#############################


@njit(f8[:](f8, f8[:], f8, f8[:], f8, f8, f8, f8[:], f8[:], f8[:, :]))
def _survive_step_plus_Lpartial_1d(kx1, Lvec1, kx2, Lvec2, kpath, odds, B, Lset, xxb, ccc):
    """ Private.

    Return the probability to survive this step as well as
    the real space distance (weighed by survival).
    Single step, so this is only valid if the resulting probability
    is substantial!

    Euler integration because I am too lazy for this code that will
    be fast enough to implement a function that finds the exact
    halfway point between phi1 and phi2. I mean halfway in k distance
    *along the surface* efficiently.

    kpath = get_pathlength_2d(phi1, phi2, xxb, ccc, err)
    is presumed to have integrity.
    This pre-computation prevents this being re-computed every time.

    odds is the change to even start this interval.
    """

    iLavg = 0.5 * (1 / np.linalg.norm(Lvec1) + 1 / np.linalg.norm(Lvec2))
    prob = np.exp(-HBAR / (abs(B) * Q) * kpath * iLavg)

    # Integral is usually written P(0) * integral(vvec * P(t) dt)
    #   where P(0)=odds are the odds to reach this sector.
    #   Here expressed in purely mean free paths by transforming t -> k basis.
    #   This is physically the contribution to the mean free path including
    #   the impact of the magnetic field.
    integral = odds * Lvec1 / np.linalg.norm(Lvec1)
    integral += odds * prob * Lvec2 / np.linalg.norm(Lvec2)
    integral *= HBAR * kpath / (Q * abs(B)) / 2
    return np.array([prob, integral[0], integral[1]])


@njit(f8[:](f8, f8[:], f8, f8[:], f8, f8[:], f8, f8, f8, f8, f8[:], f8[:], f8[:, :]))
def _survive_step_plus_Lpartial3_1d(kx1, Lvec1, kx2, Lvec2, kx3, Lvec3,
                                    kpath12, kpath23, odds, B, Lset, xxb, ccc):
    """ Private. Simpson version  """

    iLavg = 0.5 * (1 / np.linalg.norm(Lvec1) + 1 / np.linalg.norm(Lvec2))
    prob12 = np.exp(-HBAR / (abs(B) * Q) * kpath12 * iLavg)
    iLavg = 0.5 * (1 / np.linalg.norm(Lvec2) + 1 / np.linalg.norm(Lvec3))
    prob23 = np.exp(-HBAR / (abs(B) * Q) * kpath23 * iLavg)

    # Integral is usually written P(0) * integral(vvec * P(t) dt)
    #   where P(0)=odds are the odds to reach this sector.
    #   Here expressed in purely mean free paths by transforming t -> k basis.
    #   This is physically the contribution to the mean free path including
    #   the impact of the magnetic field.
    # Fit y = a*k^2 + b*k + c on the integrand (here named y for ease)
    # and integrate this fit to get Simpson on unequal spacing.
    assert(kpath12 > 0)
    assert(kpath23 > 0)
    a, b, c = fit_parabola(0, kpath12, kpath12 + kpath23,
                           Lvec1[0] / np.linalg.norm(Lvec1),
                           Lvec2[0] / np.linalg.norm(Lvec2) * prob12,
                           Lvec3[0] / np.linalg.norm(Lvec3) * prob12 * prob23)
    kp = kpath12 + kpath23
    integral_x = a * kp + .5 * b * kp**2 + 1 / 3 * c * kp**2
    integral_x *= odds * HBAR / (Q * abs(B))

    a, b, c = fit_parabola(0, kpath12, kpath12 + kpath23,
                           Lvec1[1] / np.linalg.norm(Lvec1),
                           Lvec2[1] / np.linalg.norm(Lvec2) * prob12,
                           Lvec3[1] / np.linalg.norm(Lvec3) * prob12 * prob23)
    kp = kpath12 + kpath23
    integral_y = a * kp + .5 * b * kp**2 + 1 / 3 * c * kp**2
    integral_y *= odds * HBAR / (Q * abs(B))

    return np.array([prob12 * prob23, integral_x, integral_y])


@njit(f8[:](f8, f8, f8, f8[:], f8[:], f8[:, :], f8, f8, f8, f8))
def _rec_partial_Lvec_1d(kx1, kx2, B, Lset, xxb, ccc, err, odds, int_x, int_y):
    """ Private """

    assert(abs(kx1 - kx2) > 1e-5)  # Maybe error<=1e-8 while B<=1e-6?
    if odds < err / 10:
        return np.array([0., 0, 0])

    # Analogous to 2d, just using kx rather than phi.
    half = .5 * (kx1 + kx2)
    k1h = get_pathlength_1d(kx1, half, xxb, ccc, err)
    kh2 = get_pathlength_1d(half, kx2, xxb, ccc, err)
    Lv1 = get_Lvec_1d(kx1, Lset, xxb, ccc)
    Lvh = get_Lvec_1d(half, Lset, xxb, ccc)
    Lv2 = get_Lvec_1d(kx2, Lset, xxb, ccc)

    # Standard technique: Compare Euler to Simpson for error estimate.
    # Use the Simpson result.
    prob, ix, iy = _survive_step_plus_Lpartial_1d(kx1, Lv1, kx2, Lv2, k1h + kh2,
                                                  odds, B, Lset, xxb, ccc)
    prob2, ix2, iy2 = _survive_step_plus_Lpartial3_1d(kx1, Lv1, half, Lvh,
                                                      kx2, Lv2, k1h, kh2, odds,
                                                      B, Lset, xxb, ccc)

    if abs(int_x + ix2) > abs(int_y + iy2):
        diff = abs(ix2 - ix) / abs(int_x + ix2) * 100
    else:
        diff = abs(iy2 - iy) / abs(int_y + iy2) * 100

    if diff < err and odds * abs(prob - prob2) < err / 100 and prob > 0.1:
        return np.array([prob2, ix2, iy2])
    else:
        probC, ixC, iyC = _rec_partial_Lvec_1d(kx1, half, B, Lset, xxb, ccc,
                                               err, odds, int_x, int_y)
        probD, ixD, iyD = _rec_partial_Lvec_1d(half, kx2, B, Lset, xxb, ccc,
                                               err, odds * probC,
                                               int_x + ixC, int_y + iyC)
        return np.array([probC * probD, ixC + ixD, iyC + iyD])


@njit(f8[:](f8, f8, f8, f8[:], f8[:], f8[:, :], f8))
def partial_Lvec_1d(kx1, kx2, B, Lset, xxb, ccc, err):
    """ Compute the distance a quasi-particle traverses until the next interval
    on the Fermi surface, as well as the probability to reach that point. """

    assert(kx2 > kx1)
    return _rec_partial_Lvec_1d(kx1, kx2, B, Lset, xxb, ccc, err, 1, 0, 0)


@njit(f8[:, :](f8[:], f8, f8[:], f8[:], f8[:, :], f8))
def partial_Lvecs_1d(kkx, B, Lset, xxb, ccc, err):
    """ Convenience.
    Specifically for the full pocket: kkx starts at 0 and ends at 2pi/A.
    """

    assert(abs(kkx[0]) < 1)
    assert(abs(kkx[-1] - 2 * np.pi / A) < 1)

    N = len(kkx)
    odds = np.zeros(N - 1)
    pLxx = np.zeros(N - 1)
    pLyy = np.zeros(N - 1)
    for i in range(N - 1):
        odds[i], pLxx[i], pLyy[i] = partial_Lvec_1d(kkx[i], kkx[i + 1],
                                                    B, Lset, xxb, ccc, err)

    r = np.empty((3, N - 1))
    r[0] = odds
    assert(max(odds) < 1)
    assert(min(odds) >= 0)
    r[1] = pLxx
    r[2] = pLyy
    return r


#############################
# Displacement 1d
# Conductivity
#############################


def sigma_2d(B, N, Lset, xxb, ccc, err):
    """ Get the conductivity using this FS and these settings.
    No error estimates. Just get it done.

    !! Do not trust the sign of sxy. Judge for yourself.

    Returns sxx, sxy, syy
    The latter to keep output the same as 1d, it should be
    the same within error as sxx.
    """

    assert(N % 2)
    pphi = np.linspace(0, 2 * np.pi, N)
    odds, pLxx, pLyy = partial_Lvecs_2d(pphi, B, Lset, xxb, ccc, err)
    Lxx, Lyy = convolve_partial_Lvecs(odds, pLxx, pLyy)

    sxx = 0
    sxy = 0
    syy = 0
    for i in range(N - 1):
        w = 2 + (i % 2) * 2
        Ldir = get_Ldir_2d(pphi[i], xxb, ccc)
        dk = get_pathlength_2d(pphi[i], pphi[i + 1], xxb, ccc, err)
        sxx += Ldir[0] * Lxx[i] * dk * w / 3
        sxy += Ldir[0] * Lyy[i] * dk * w / 3
        syy += Ldir[1] * Lyy[i] * dk * w / 3
    sxx *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    sxy *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    syy *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    assert(w == 4)

    return sxx, sxy * (-1 + 2 * (B >= 0)), syy


def sigma_1d(B, N, Lset, xxb, ccc, err):
    """ Get the conductivity using this FS and these settings.
    No error estimates. Just get it done.

    !! Returns the conductivity for the full FS,
        so BOTH sheets, even though xxb, ccc only
        parameterises 1, inversion symmetry is assumed.

    !! Do not trust the sign of sxy, it is arbitrary.
        Only aspect guaranteed is that the sign of B flips its sign.
    """

    assert(N % 2)
    kkx = np.linspace(0, 2 * np.pi / A, N)
    odds, pLxx, pLyy = partial_Lvecs_1d(kkx, B, Lset, xxb, ccc, err)
    Lxx, Lyy = convolve_partial_Lvecs(odds, pLxx, pLyy)

    sxx = 0
    sxy = 0
    syy = 0
    for i in range(N - 1):
        w = 2 + (i % 2) * 2
        Ldir = get_Ldir_1d(kkx[i], xxb, ccc)
        dk = get_pathlength_1d(kkx[i], kkx[i + 1], xxb, ccc, err)
        sxx += Ldir[0] * Lxx[i] * dk * w / 3
        sxy += Ldir[0] * Lyy[i] * dk * w / 3
        syy += Ldir[1] * Lyy[i] * dk * w / 3

    sxx *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    sxy *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    syy *= Q**2 / (4 * np.pi**3 * HBAR) * 2 * np.pi / C
    assert(w == 4)

    # Factor 2 for sheets.
    return 2 * sxx, 2 * sxy * (-1 + 2 * (B >= 0)), 2 * syy


def sigma_with_error(B, N, Lset, xxb, ccc, err, is_2d, *, loud=False):
    """ Vary N and err by 2x to estimate the error. Combines 1d & 2d


    Give a string to loud if you wish to provide info to the user
    such as which band is used.
    """

    if B == 0:
        raise ValueError('Magnetic field 0 is forbidden. Use 1e-5 or 1e-6 T.')

    sigma = sigma_2d if is_2d else sigma_1d

    res = sigma(B, N, Lset, xxb, ccc, err)
    test1 = sigma(B, N // 2 + 1, Lset, xxb, ccc, err)
    test2 = sigma(B, N, Lset, xxb, ccc, err * 2)

    err1 = max(abs(test1[0] - res[0]), abs(test2[0] - res[0]))
    err2 = max(abs(test1[1] - res[1]), abs(test2[1] - res[1]))
    err3 = max(abs(test1[2] - res[2]), abs(test2[2] - res[2]))

    if loud != False:
        extra = '' if loud == True else f' ({loud})'
        err_yy_N = abs(test1[0] - res[0]) / res[0]
        err_yy_e = abs(test2[0] - res[0]) / res[0]
        print(f'\nB={B} sxx: Error from N is {err_yy_N:.1e}, '
              f'error from err is {err_yy_e:.1e}{extra}')

        err_yy_N = abs(test1[1] - res[1]) / res[1]
        err_yy_e = abs(test2[1] - res[1]) / res[1]
        print(f'B={B} sxy: Error from N is {err_yy_N:.1e}, '
              f'error from err is {err_yy_e:.1e}{extra}')

        err_yy_N = abs(test1[2] - res[2]) / res[2]
        err_yy_e = abs(test2[2] - res[2]) / res[2]
        print(f'B={B} syy: Error from N is {err_yy_N:.1e}, '
              f'error from err is {err_yy_e:.1e}{extra}\n', flush=True)

    return tuple(res), tuple([err1, err2, err3])


def sigma_auto(B, N, Lset, name, err, *, loud=False):
    """ sigma_with_error but using the name and doing smoothing internally.

    Loud: boolean
    """

    try:
        xxb, ccc = smooth_fs_1d(name)
        is_2d = False
    except Exception:
        xxb, ccc = smooth_fs_2d(name)
        is_2d = True

    if loud:
        loud = name
    ss, err = sigma_with_error(B, N, Lset, xxb, ccc, err, is_2d, loud=loud)
    return ss, err


def convert_rho(ss, err):
    """ Given [sxx, sxy, syy] and errors, get [rxx, ryx, ryy] and errors. """

    assert(len(ss) == 3)
    assert(len(err) == 3)
    ss = np.array(ss)
    err = np.array(err)

    # d for denominator
    d = ss[0] * ss[2] + ss[1]**2
    rxx = ss[2] / d
    ryx = ss[1] / d
    ryy = ss[0] / d

    # Error propagation following delta_y^2 = (dy/dx)^2 delta_x
    #   adding to delta_y^2 (the variance) for each x that exists
    #   in the conversion formula y(x1, x2, x3).
    #   Here, y is one of the resistivities, x's are sxx, sxy, syy.
    exx = (-ss[2]**2 / d**2)**2 * err[0]**2
    exx += (-2 * ss[2] * ss[1] / d**2)**2 * err[1]**2
    exx += (1 / d - ss[0] * ss[2] / d**2)**2 * err[2]**2
    exx = np.sqrt(exx)

    eyx = (-ss[1] / d**2 * ss[2])**2 * err[0]**2
    eyx += (1 / d - 2 * ss[1]**2 / d**2)**2 * err[1]**2
    eyx += (-ss[1] / d**2 * ss[0])**2 * err[2]**2
    eyx = np.sqrt(eyx)

    eyy = (1 / d - ss[0] * ss[2] / d**2)**2 * err[0]**2
    eyy += (-2 * ss[0] * ss[1] / d**2)**2 * err[1]**2
    eyy += (-ss[0]**2 / d**2)**2 * err[2]**2
    eyy = np.sqrt(eyy)

    return tuple([tuple([rxx, ryx, ryy]), tuple([exx, eyx, eyy])])


print(f'Imported compute.py in {time.process_time() - _st:.1f} s', flush=True)