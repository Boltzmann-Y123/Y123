# 05-04-2024
#   Apply the smoothing of 02.py to the Fermi surface data.
# Author: Roemer Hinlopen
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

# See Buckley1991
# b-axis is the direction of the chain as per usual convention
A = 3.821e-10
B = 3.883e-10
plt.rc('font', size=20)
figfolder = '04 figs'


def _import_fs_data_file(name):
    """ Names are string. 'chain', 'inner_square', 'outer_square' or 'S' """

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

    ! Assumes the data is periodic in 2pi
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

    See 02.py for the procedure details.
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

    The first returned value are the x intervals in which the 
    data is segmented. The second are for each interval the 
    polynomial coefficients to obtain yy (the FS radius)
    """

    assert(len(xx) > 5)
    x_ext = list(xx[-5:] - modx) + list(xx) + list(xx[:5] + modx)
    y_ext = list(yy[-5:]) + list(yy) + list(yy[:5])
    Ppoly = si.CubicSpline(x_ext, y_ext)
    return Ppoly.x[5:-5], Ppoly.c[:, 5:-4]


def eval_ppoly(x, xxbreak, cccoeff, modx):
    """ Evaluate this interpolation at x.

    Assumes xxbreak and cccoeff are made with make_stripped_int()
    Feel free to provide x negative / above modx.
    """

    # N is number of breakpoints
    # also N intervals (N-1 normal ones, and 1 for cyclic)
    N = len(xxbreak)
    x %= modx
    maxi = N - 1
    mini = 0
    while maxi - mini > 1:
        middle = mini + (maxi - mini) // 2
        xmiddle = xxbreak[middle]
        if xmiddle - x > 0:
            maxi = middle
        else:
            mini = middle
    index_below = maxi
    while x < xxbreak[index_below]:
        index_below -= 1

    cc = cccoeff[:, index_below]
    order = len(cc) - 1
    y_eval = 0
    for i, c in enumerate(cc):
        y_eval += c * (x - xxbreak[index_below])**(order - i)
    return y_eval


def differentiate_coeff(cccoeff):
    """ Get the ppoly coefficients for the derivative. """

    order = len(cccoeff) - 1
    ddderiv = np.zeros([order, len(cccoeff[0])])
    for i in range(order):
        ddderiv[i, :] = cccoeff[i, :] * (order - i)
    return ddderiv


def smooth_fs_2d(name):
    """ Obtain a series of phi intervals with a 3rd degree polynomial
    defining the FS on each. Continuous value & 1st derivative. """

    if 'chain' in name:
        raise ValueError('Only 2d Fermi surface sheets.')

    xx, yy = complete_fs_data(name)
    rr = np.sqrt(xx**2 + yy**2)
    pphi = np.arccos(xx / np.sqrt(xx**2 + yy**2))
    pphi[yy < 0] *= -1
    pphi %= 2 * np.pi
    order = np.argsort(pphi)
    rr = rr[order]
    pphi = pphi[order]

    # Add a number of out-of-bound points to ensure smoothing through 0 & 2pi.
    # Choose at least half the initial-smooth order (9//2 at the time of writing.)
    oob = 10
    pphi = list(pphi[-oob:] - 2 * np.pi) + list(pphi) + list(pphi[:oob] + 2 * np.pi)
    pphi = np.array(pphi)
    rr = list(rr[-oob:]) + list(rr) + list(rr[:oob])
    rr = np.array(rr)
    rr2 = smooth(pphi, rr, 9, 1, 2 * np.pi)

    # Do a second round of smoothing on the derivative. See 02.py.
    # The mean free path / velocity direction must be smooth.
    pphi3, rr3 = stabilised_deriv(pphi[oob:-oob], rr2[oob:-oob], 1000, 31, 2 * np.pi)
    xxb, ccc = make_stripped_cyclic_interpolation(pphi3, rr3, 2 * np.pi)
    return xxb, ccc


def show_smooth_fs_2d(name, xxb, ccc):
    """ Visualise the digitized and smoothed FS and Ldir across. 

    Use an isotropic-tau view with constant mass all around.
    Reason is that an isotropic L will always look like a perfect 
        circle, whereas this view shows the true rotations
        and remaining derivative jitter much better.
        Calculations use isotropic L.
    See Ong1991 for the relation between L orbits and Hall effect.
    """

    phi_plt = np.linspace(-1, 2 * np.pi + 1, 10000)
    r_plt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi) for x in phi_plt])
    ddd = differentiate_coeff(ccc)
    dplt = [eval_ppoly(phi, xxb, ddd, 2 * np.pi) for phi in phi_plt]

    xx, yy = complete_fs_data(name)
    plt.figure(f'fs {name}', figsize=(10, 8))
    plt.scatter(xx, yy, color='tab:blue')
    plt.plot(r_plt * np.cos(phi_plt), r_plt * np.sin(phi_plt), color='tab:red')
    plt.gca().set_aspect('equal')
    plt.xlabel('$k_x$ (1/m)')
    plt.ylabel('$k_y$ (1/m)')
    plt.title(f'2d {name} pocket smoothed')

    if name == 'inner_square':
        plt.savefig(f'{figfolder}/10 inner square pocket smoothed.png', dpi=300)
    else:
        plt.savefig(f'{figfolder}/12 outer square pocket smoothed.png', dpi=300)
        assert(name == 'outer_square')


    LL = []
    for i in range(0, len(dplt), len(dplt) // 500):
        phi = phi_plt[i]
        if phi >= 0 and phi < 2 * np.pi:
            Ldir = np.array([np.cos(phi), np.sin(phi)])
            Ldir -= np.array([-np.sin(phi), np.cos(phi)]) * dplt[i] / r_plt[i]

            Ldir /= np.linalg.norm(Ldir)
            Ldir *= 1.05e-34 * r_plt[i] / (5 * 9.109e-31) * 1e-13

            pp = np.array([r_plt[i] * np.cos(phi), r_plt[i] * np.sin(phi)])
            plt.plot([pp[0], pp[0] + Ldir[0]], [pp[1], pp[1] + Ldir[1]], color='black')
            LL.append(Ldir)

    Lxx, Lyy = np.array(LL).T
    plt.figure(f'L orbit 2d {name}: hkt/m', figsize=(10, 8))
    plt.title(f'L orbit for {name} pocket, isotropic tau (0.1 ps)')
    plt.plot(Lxx * 1e9, Lyy * 1e9)
    plt.xlabel('$l_x$ (nm)')
    plt.ylabel('$l_y$ (nm)')
    plt.gca().set_aspect('equal')

    if name == 'inner_square':
        plt.savefig(f'{figfolder}/11 inner square L orbit isotropic tau.png', dpi=300)
    else:
        plt.savefig(f'{figfolder}/13 outer square L orbit isotropic tau.png', dpi=300)
        assert(name == 'outer_square')


def smooth_fs_1d():
    """ See 2d. Now instead of (phi, r), smooth in (kx, ky) """

    modx = 2 * np.pi / A
    xx, yy = complete_fs_data('chain')

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


def show_smooth_fs_1d(xxb, ccc):
    """ See 2d. See Ong1991 for the relation between L orbits and Hall effect. """

    xx, yy = complete_fs_data('chain')

    xplt = np.linspace(-np.pi / A, np.pi / A, 10000)
    yplt = np.array([eval_ppoly(x, xxb, ccc, 2 * np.pi / A) for x in xplt])

    plt.figure('Chain', figsize=(14, 8))
    plt.scatter(xx, yy, color='tab:blue')
    plt.plot(xplt, yplt, color='tab:red')
    plt.gca().set_aspect('equal')
    plt.xlabel('$k_x$ (1/m)')
    plt.ylabel('$k_y$ (1/m)')
    plt.title('Chain pocket smoothed')
    plt.savefig(f'{figfolder}/14 1d pocket smoothed.png', dpi=300)

    LL = []
    ddd = differentiate_coeff(ccc)
    for i in range(0, len(xplt), len(xplt) // 250):
        if xplt[i] >= -np.pi / A and xplt[i] <= np.pi / A:
            d = eval_ppoly(xplt[i], xxb, ddd, 2 * np.pi / A)
            Ldir = np.array([d, 1])

            Ldir /= np.linalg.norm(Ldir)
            Ldir *= 1.05e-34 * abs(yplt[i]) / (5 * 9.109e-31) * 1e-13

            plt.plot([xplt[i], xplt[i] + Ldir[0]], [yplt[i], yplt[i] + Ldir[1]], color='black')
            LL.append(Ldir)

    Lxx, Lyy = np.array(LL).T
    plt.figure('L orbit 1d', figsize=(14, 8))
    plt.title('isotropic-tau (0.1 ps) L orbit of the 1d pocket')
    plt.plot(Lxx * 1e9, Lyy * 1e9)
    plt.xlabel('$l_x$ (nm)')
    plt.ylabel('$l_y$ (nm)')
    plt.gca().set_aspect('equal')
    plt.savefig(f'{figfolder}/15 1d pocket L orbit isotropic tau.png', dpi=300)


xxb, ccc = smooth_fs_2d('inner_square')
show_smooth_fs_2d('inner_square', xxb, ccc)
xxb, ccc = smooth_fs_2d('outer_square')
show_smooth_fs_2d('outer_square', xxb, ccc)
xxb, ccc = smooth_fs_1d()
show_smooth_fs_1d(xxb, ccc)


plt.show()
