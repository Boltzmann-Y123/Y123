# Code specifically dedicated to extract the true underlying plane carrier density of the model
#   Carrington2007 evaluated optimal doping, but that will be modified by smoothing
#   and the fact we are ignoring kz.
#   Result is still 0.157 in the end, showing how subtle the changes by smoothing are.
# Author: Roemer Hinlopen

import numpy as np
import matplotlib.pyplot as plt

code = __import__('04_compute')


def create_larger_2d_fs(*, show=False):
    """ The outer square BEFORE hybridisation. So larger than the
    outer_square fs taking a part of the chain. This must be
    created to compute the full planar carrier density w/o chains.

    Returns raw (pphi, rr) values.
    Show highlights in black exactly what is included in
        the planar carrier density.
    """

    xxbO, cccO = code.smooth_fs_2d('outer_square')
    xxbI, cccI = code.smooth_fs_2d('inner_square')
    xxbC, cccC = code.smooth_fs_1d('chain')

    pphi = np.linspace(0, 2 * np.pi, 1000)
    rrO = []
    rrI = []
    for phi in pphi:
        r = code.get_r_2d(phi, xxbO, cccO)
        rrO.append(r)
        r = code.get_r_2d(phi, xxbI, cccI)
        rrI.append(r)
    rrO = np.array(rrO)
    rrI = np.array(rrI)

    kkx = np.linspace(-np.pi / code.A, np.pi / code.A, 1000)
    kky = []
    for kx in kkx:
        ky = code.get_ky_1d(kx, xxbC, cccC)
        kky.append(ky)
    kky = np.array(kky)

    if show:
        plt.figure(figsize=(10, 8))
        plt.title('Defining the pre-hybridisation FS to get n_pl')
        plt.scatter(rrO * np.cos(pphi), rrO * np.sin(pphi), label='outer 2d')
        plt.scatter(kkx, kky, color='tab:green')
        plt.scatter(kkx, -kky, color='tab:green', label='1d')

        # Inner is included in the count directly, no hybridisation to care about.
        plt.scatter(rrI * np.cos(pphi), rrI * np.sin(pphi), color='black')
        plt.xlabel('$k_x$ (1/m)')
        plt.ylabel('$k_y$ (1/m)')
        plt.gca().set_aspect('equal')

    ref = 1.04
    refx = 3.5e9

    # Northeast on outer_square
    part1 = pphi < ref
    pphi_big = list(pphi[part1])
    rr_big = list(rrO[part1])

    # North on chain
    part2 = (kkx < refx) & (kkx > -refx)
    kkx = kkx[part2]
    kky = np.abs(kky[part2])
    rr_temp = np.sqrt(kkx**2 + kky**2)
    phi_temp = np.arccos(kkx / rr_temp)
    pphi_big += list(phi_temp)
    rr_big += list(rr_temp)

    # West on outer square
    part3 = (pphi > np.pi - ref) & (pphi < np.pi + ref)
    pphi_big += list(pphi[part3])
    rr_big += list(rrO[part3])

    # South on chain
    part4 = (kkx < refx) & (kkx > -refx)
    kkx = kkx[part4]
    kky = -np.abs(kky[part4])
    rr_temp = np.sqrt(kkx**2 + kky**2)
    phi_temp = 2 * np.pi - np.arccos(kkx / rr_temp)
    pphi_big += list(phi_temp)
    rr_big += list(rr_temp)

    # Southeast on outer square
    part5 = pphi > 2 * np.pi - ref
    pphi_big += list(pphi[part5])
    rr_big += list(rrO[part5])

    pphi_big = np.array(pphi_big)
    rr_big = np.array(rr_big)

    if show:
        plt.scatter(rr_big * np.cos(pphi_big),
                    rr_big * np.sin(pphi_big),
                    color='black', label='pre-hybridisation 2d')
        plt.legend(loc='upper right')
        plt.subplots_adjust(right=0.8)

    order = np.argsort(pphi_big)
    return pphi_big[order], rr_big[order]


def direct_carrier_density(pphi, rr):
    """ """

    for a, b in zip(pphi[1:], pphi[:-1]):
        assert(a > b)
    assert(pphi[0] == 0)
    assert(pphi[-1] == 2 * np.pi)

    xx = rr * np.cos(pphi)
    yy = rr * np.sin(pphi)
    area = 0
    for i in range(len(pphi) - 1):
        area += xx[i] * yy[i + 1] - xx[i + 1] * yy[i + 1]
    area = abs(area)

    # Spin degeneracy included
    n = area / (4 * np.pi**3) * 2 * np.pi / code.C
    return n


def get_bad_planar_carrier_density():
    """ The inner plane, and the outer plane post-hybridisation. """

    xxbO, cccO = code.smooth_fs_2d('inner_square')
    pphiO = np.linspace(0, 2 * np.pi, 1000)
    rrO = []
    for phi in pphiO:
        r = code.get_r_2d(phi, xxbO, cccO)
        rrO.append(r)
    rrO = np.array(rrO)
    outer = direct_carrier_density(pphiO, rrO)

    xxbI, cccI = code.smooth_fs_2d('inner_square')
    pphiI = np.linspace(0, 2 * np.pi, 1000)
    rrI = []
    for phi in pphiI:
        r = code.get_r_2d(phi, xxbI, cccI)
        rrI.append(r)
    rrI = np.array(rrI)
    inner = direct_carrier_density(pphiI, rrI)
    return inner + outer


def get_planar_carrier_density():
    """ The inner plane, and the outer plane pre-hybridisation. """

    pphi, rr = create_larger_2d_fs(show=False)
    outer_pre_hybrid = direct_carrier_density(pphi, rr)

    xxbI, cccI = code.smooth_fs_2d('inner_square')
    pphiI = np.linspace(0, 2 * np.pi, 1000)
    rrI = []
    for phi in pphiI:
        r = code.get_r_2d(phi, xxbI, cccI)
        rrI.append(r)
    rrI = np.array(rrI)
    inner = direct_carrier_density(pphiI, rrI)
    return inner + outer_pre_hybrid


if __name__ == '__main__':
    plt.rc('font', size=20)
    create_larger_2d_fs(show=True)
    plt.savefig('04 figs/20 define pre-hybridisation FS.png', dpi=300)

    n = get_planar_carrier_density()

    # Full band is 2, spin deg is included in n
    with open('04 figs/21 carrier density.txt', 'w') as f:
        p = n * code.A * code.B * code.C / 2
        print(f'Full planar carrier density is {n:.3e} 1/m^3')
        f.write(f'Full planar carrier density is {n:.3e} 1/m^3\n')
        print(f'Equivalent to p = {p - 1:.3f} [memory: on author\'s machine 0.157]')
        f.write(f'Equivalent to p = {p - 1:.3f} [memory: on author\'s machine 0.157]\n')

        n = get_bad_planar_carrier_density()
        p = n * code.A * code.B * code.C / 2
        print(f'(If hybridisation is not bypassed I get p = {p - 1:.3f})')
        f.write(f'(If hybridisation is not bypassed I get p = {p - 1:.3f})\n')

    print(flush=True)
    plt.show()
