# 03-04-2024 
# Instead of cylindrical harmonics which do not fit well,
#   do a direct smoothing and interpolation of the data.
#   This is a bit tricky because the derivative must be smooth
#   for the velocity vectors.
#
# Files 02.py and 03.py show step by step how this new method was developed. 
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
figsfolder= '04 figs'


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
    else:
        raise ValueError(f'No fs completion implemented for "{name}"')

    return xx, yy


def plot_fs_data():
    """ Just show the digitization data from Carrington2007. 

    No manipulation. 
    Do include mirrors in the plot to show the full FS.
    """

    plt.figure('fs data', figsize=(10, 8))
    names = ['chain', 'inner_square', 'outer_square', 'S']
    cc = plt.get_cmap('tab10').colors
    for name, c in zip(names, cc):
        xx, yy = complete_fs_data(name)
        plt.scatter(xx, yy, s=20, color=c)
        if name == 'chain':
            plt.scatter(xx, -yy, s=20, color=c)

    piA = np.pi / A
    piB = np.pi / B
    plt.plot([-piA, piA], [-piB] * 2, color='black', lw=8)
    plt.plot([-piA] * 2, [-piB, piB], color='black', lw=8)
    plt.plot([-piA, piA], [piB] * 2, color='black', lw=8)
    plt.plot([piA] * 2, [-piB, piB], color='black', lw=8)
    plt.xlabel('$k_x$ (1/m)')
    plt.ylabel('$k_y$ (1/m)')
    plt.title('Raw digitized from Carrington2007')
    plt.savefig(f'{figsfolder}/01 Raw FS data (kx,ky).png', dpi=300)


def radial_plot_data():
    """ Explore the phi basis for the FS description for smoothing of closed pockets. """

    plt.figure('fs radial', figsize=(10, 8))
    plt.title('Raw digitized in radial basis')

    names = ['chain', 'inner_square', 'outer_square', 'S']
    cc = plt.get_cmap('tab10').colors
    for name, c in zip(names, cc):
        xx, yy = complete_fs_data(name)

        rr = np.sqrt(xx**2 + yy**2)
        pphi = np.arccos(xx / rr)
        pphi[yy < 0] *= -1
        pphi %= 2 * np.pi

        plt.scatter(pphi, rr, s=20, color=c)
    
    plt.xlabel('$\\phi$ (rad)')
    plt.ylabel('$k_r$ (1/m)')
    plt.xlim(0, 2 * np.pi)
    plt.savefig(f'{figsfolder}/02 Raw FS data radial.png', dpi=300)


def smooth(xx, yy, order, poly_order, modx):
    """ Can be a bit slow if you give a lot of data.

    Basically fit a polynomial on 'order' points nearby,
    must be odd. Then evaluate the polynomial at the
    central position and return the resulting y* value.

    A fast implementation would do the polynomial fits with
    a coefficient table or matrix, and update it point-to-point.
    But this is fast enough for a few hundred points and keeps it simple.

    ! Assumes the data is periodic.
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
    # The first few fits require the cyclic nature of x
    #   to translate the last points in array over by modx.
    for i in range(order // 2):
        xnow = list(xx[i - order // 2:] - modx) + list(xx[:i + order // 2 + 1])
        assert(len(xnow) == order)
        ynow = list(yy[i - order // 2:]) + list(yy[:i + order // 2 + 1])
        pp = np.polyfit(xnow, ynow, deg=poly_order)
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    # The middle is straightforward
    for i in range(order // 2, N - order // 2):
        xnow = xx[i - order // 2:i + order // 2 + 1]
        assert(len(xnow) == order)
        ynow = yy[i - order // 2:i + order // 2 + 1]
        pp = np.polyfit(xnow, ynow, deg=poly_order)
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    # The end requires again the cyclic nature of x to translate the points over.
    for i in range(N - order // 2, N):
        xnow = list(xx[i - order // 2:] - modx) + list(xx[:i - N + order // 2 + 1])
        assert(len(xnow) == order)
        ynow = list(yy[i - order // 2:]) + list(yy[:i - N + order // 2 + 1])
        new_y[i] = sum([p * xx[i]**(poly_order - j) for j, p in enumerate(pp)])

    return new_y


def test_smoothing_simple():
    """ Use few points and low order to show how this works. """

    xx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    yy = np.array([1, 2, 3, 4, 5, 4, 3, 2.5, 2, 1])
    yy2 = smooth(xx, yy, 3, 1, 10)

    plt.figure(figsize=(10, 8))
    plt.scatter(xx, yy)
    plt.plot(xx, yy2)
    plt.title('Raw (dots) and linear cyclic smooth (line)')


def test_smoothing_noise():
    """ Test smoothing with random noise on both x and y. """

    xx = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    yy = np.sin(xx)

    xx += np.random.random(len(xx)) * 0.2 * (xx[1] - xx[0])
    yy += np.random.random(len(yy)) * 0.05

    plt.figure('#1: Smoothing noise directly', figsize=(10, 8))
    plt.scatter(xx, yy, label='Raw')
    yy2 = smooth(xx, yy, 9, 1, 2 * np.pi)
    # plt.plot(xx, yy2, label='Smoothed')

    xfeed = list(xx - 2 * np.pi) + list(xx) + list(xx + 2 * np.pi)
    yy3 = list(yy2) * 3
    Interp = si.CubicSpline(xfeed, yy3)

    xplt = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    plt.plot(xplt, Interp(xplt), label='Smoothed + interpolated', lw=3)
    plt.plot(xplt, np.sin(xplt), label='True', lw=3)
    plt.legend()
    plt.savefig(f'{figsfolder}/03 Step 1 - Smoothing noise directly.png', dpi=300)


def test_smoothing_deriv():
    """ Take a derivative, smooth it, then integrate it. """

    # Copy of smoothing_noise
    xx = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    yy = np.sin(xx)
    xx += np.random.random(len(xx)) * 0.2 * (xx[1] - xx[0])
    yy += np.random.random(len(yy)) * 0.05

    yy2 = smooth(xx, yy, 9, 1, 2 * np.pi)

    xfeed = list(xx - 2 * np.pi) + list(xx) + list(xx + 2 * np.pi)
    yy3 = list(yy2) * 3
    Interp = si.CubicSpline(xfeed, yy3)

    # Now go for the derivative.
    xplt = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    dyy = (Interp(xplt + 0.001) - Interp(xplt)) / 0.001

    plt.figure('#2: Smooth derivative', figsize=(10, 8))
    plt.plot(xplt, np.cos(xplt), label='True', lw=3)
    plt.plot(xplt, dyy, label='Deriv of #1', lw=3)

    stab_dyy = smooth(xplt, dyy, 61, 1, 2 * np.pi)
    stab_dyy += np.mean(dyy) - np.mean(stab_dyy)
    plt.plot(xplt, stab_dyy, label='Smoothed deriv', lw=3)
    plt.legend()
    plt.savefig(f'{figsfolder}/04 Step2 - Smoothing derivative.png', dpi=300)

    # Step 5: Integrate stabilised derivative
    #   Half the (arbitrarily long!) length to use Simpson's rule
    stabilised_xx = xplt[::2]
    stabilised_yy = np.zeros(len(stab_dyy) // 2)

    for i in range(1, len(stabilised_xx)):
        delta = stab_dyy[2 * i] + 2 * stab_dyy[(2 * i + 1)  % len(stab_dyy)] + stab_dyy[(2 * i + 2) % len(stab_dyy)]
        delta *= 0.5 * (xplt[1] - xplt[0])
        stabilised_yy[i] = stabilised_yy[i - 1] + delta
    stabilised_yy += np.mean(Interp(xplt)) - np.mean(stabilised_yy)

    plt.figure('#3: Integrated', figsize=(10, 8))
    plt.scatter(xx, yy, label='Noisy original')
    plt.plot(xplt, np.sin(xplt), label='True', lw=3)
    plt.plot(stabilised_xx, stabilised_yy, label='Deriv smoothed', lw=3)
    plt.legend()
    plt.savefig(f'{figsfolder}/05 Step 3 - integrate.png', dpi=300)


# Step 1: Show data
plot_fs_data()
radial_plot_data()

# Step 2: Set up smoothing with periodic boundaries
test_smoothing_simple()

# Step 3: Smoothing the derivative
test_smoothing_noise()
test_smoothing_deriv()

plt.show()
