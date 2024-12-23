# Adjusted to show anisotropy scans rather than field scans.
#
# Show_combination(1) generates the figure in the SI of the paper.
# Author: Roemer Hinlopen

import os
import numpy as np
import matplotlib.pyplot as plt

# This unfortunately takes about 10-20 seconds
# because all functions with signature are compiled by numba.
code = __import__('04_compute')
planar = __import__('04_planar_density')


def get_path(index):
    """ Responsible for: Where is this data exactly.

    Raises if does not exist.
    """

    direc = '03 output ani sweep'
    if not os.path.isdir(direc):
        raise IOError('No output directory.')

    path = os.path.join(direc, f'{index:04d}.dat')
    if not os.path.isfile(path):
        raise IOError(f'No output file #{index:04d}')
    return path


def download(index):
    """ Get the magnetic fields, and for each the conductivities and errors. """

    path = get_path(index)
    aani = []
    sss = []
    eee = []
    with open(path) as f:
        for line in f:
            if 'sxxO' in line:
                break
        else:
            raise IOError(f'No labelline in {path}')

        for line in f:
            if not line:
                continue

            nums = [float(x) for x in line.split()]
            assert(len(nums) == 19)
            aani.append(nums[0])
            sss.append(nums[1::2])
            eee.append(nums[2::2])
            assert(len(eee[-1]) == len(sss[-1]))

    aani = np.array(aani)
    wh = aani <= ANI_MAX
    return np.array(aani)[wh], np.array(sss)[wh], np.array(eee)[wh]


def full_conversion(sss, eee):
    """ Adds the necessary conductivities and converts to
    total resistivities including error propagation.

    sss, eee: arrays (N, 9)

    Returns rrr, errr of shape (N, 3)
    where the 3 are xx, xy and yy
    Returned in muOhmcm
    """

    assert(len(sss) == len(eee))
    assert(len(np.shape(sss)) == 2)
    assert(len(np.shape(eee)) == 2)
    assert(np.shape(sss)[1] == 9)
    assert(np.shape(eee)[1] == 9)

    rrr = []
    errr = []
    for ss, ee in zip(sss, eee):
        sxx_tot = ss[0] + ss[3] + ss[6]
        sxy_tot = ss[1] + ss[4] + ss[7]
        syy_tot = ss[2] + ss[5] + ss[8]

        exx_tot = np.sqrt(ee[0]**2 + ee[3]**2 + ee[6]**2)
        exy_tot = np.sqrt(ee[1]**2 + ee[4]**2 + ee[7]**2)
        eyy_tot = np.sqrt(ee[2]**2 + ee[5]**2 + ee[8]**2)

        rr, ee = code.convert_rho([sxx_tot, sxy_tot, syy_tot],
                                  [exx_tot, exy_tot, eyy_tot])
        rrr.append(rr)
        errr.append(ee)

    return np.array(rrr) * 1e8, np.array(errr) * 1e8


def show_all_sigmas(aani, sss, eee, *, ax='new'):
    """ Convenience.

    This is mostly used as a check to explain results,
    not to visualise it for a final graph. """

    sss = np.array(sss) / 1e8
    eee = np.array(eee) / 1e8

    ms = 6

    if ax == 'new':
        f, ax = plt.subplots(figsize=(10, 5))
        f.subplots_adjust(left=0.1, bottom=0.15, right=0.85, top=0.97)

    ax.errorbar(aani, sss[:, 0], yerr=eee[:, 0], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, O}$')
    ax.errorbar(aani, sss[:, 1], yerr=eee[:, 1], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, O}$')
    ax.errorbar(aani, sss[:, 2], yerr=eee[:, 2], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, O}$')

    ax.errorbar(aani, sss[:, 3], yerr=eee[:, 3], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, I}$')
    ax.errorbar(aani, sss[:, 4], yerr=eee[:, 4], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, I}$')
    ax.errorbar(aani, sss[:, 5], yerr=eee[:, 5], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, I}$')

    ax.errorbar(aani, sss[:, 6], yerr=eee[:, 6], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, C}$')
    ax.errorbar(aani, sss[:, 7], yerr=eee[:, 7], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, C}$')
    ax.errorbar(aani, sss[:, 8], yerr=eee[:, 8], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, C}$')

    ax.set_xlim(0, ANI_MAX)
    ax.set_xlabel('$l_{1d}/l_{2d}$')
    ax.set_ylabel('$\u03C3$ (1/\u03BC\u03A9cm)')
    ax.legend(bbox_to_anchor=[1, 1], loc='upper left', frameon=False)


def show_resistivity(aani, rrr, errr, *, ax='new'):
    """ In here to easily turn on/off and keep overview """

    if ax == 'new':
        f, ax = plt.subplots()

    ax.errorbar(aani, rrr[:, 0], yerr=errr[:, 0], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{aa}$')
    ax.errorbar(aani, rrr[:, 2], yerr=errr[:, 2], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{bb}$')
    ax.errorbar(aani, rrr[:, 1], yerr=errr[:, 1], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{ba}$')
    ax.set_xlim(0, ANI_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$l_{1d}/l_{2d}$')
    ax.set_ylabel('$\u03C1$ (\u03BC\u03A9cm)')
    ax.legend(frameon=False, bbox_to_anchor=[1.0, 0.8], loc='upper right')


def show_nH_ani(aani, rrr, errr, constB, *, axA='new', axN='new'):
    """ Show n, nH and nH/ani. The true result. """

    if axA == 'new':
        f, axA = plt.subplots()

    aani_r = rrr[:, 0] / rrr[:, 2]
    # Error propagation like in convert_rho()
    eaani_r = (1 / rrr[:, 2])**2 * errr[:, 0]**2
    eaani_r += (-rrr[:, 0] / rrr[:, 2]**2)**2 * errr[:, 2]**2
    eaani_r = np.sqrt(eaani_r)

    axA.errorbar(aani, aani_r, yerr=eaani_r, lw=4,
                 color='tab:red', marker='o', ms=6)
    axA.scatter(aani, aani_r, s=50, color='tab:red')
    axA.set_xlim(0, ANI_MAX)
    axA.set_ylim(bottom=1, top=max(aani_r) + 0.25)
    axA.set_xlabel('$l_{1d}/l_{2d}$')
    axA.set_ylabel('$\u03C1_{aa}/\u03C1_{bb}$')

    if axN == 'new':
        f, axN = plt.subplots()
    nBZ = 1 / code.A / code.C / code.B
    n2d = planar.get_planar_carrier_density()
    # planar.create_larger_2d_fs(show=True)
    print(f'(pre-hybridise) planar doping per CuO2: p={n2d / nBZ / 2 - 1:.3f}')

    axN.plot([0, max(aani) * 2], [n2d] * 2, lw=2,
             dashes=[6, 2], color='black', label='n$_{pl}$')

    # Again error propagation
    RH = rrr[:, 1] / constB * 10
    eRH = errr[:, 1] / constB * 10
    nH = 1e9 / RH / code.Q
    enH = np.sqrt((-1e9 / RH**2 / code.Q)**2 * eRH**2)
    nHf = nH / aani_r
    enHf = (1 / aani_r)**2 * enH**2
    enHf += (-nH / aani_r**2)**2 * eaani_r**2
    enHf = np.sqrt(enHf)

    axN.errorbar(aani, nH, yerr=enH, color='.4', elinewidth=2,
                 label='$n_H$', lw=4, marker='o', ms=6)
    axN.errorbar(aani, nHf, yerr=enHf, lw=4, marker='o', ms=6, elinewidth=2,
                 color='tab:blue', label="$n_H\u03C1_{bb}/\u03C1_{aa}$")
    axN.legend(frameon=False, bbox_to_anchor=[1.0, 0.8], loc='upper right')
    axN.set_xlim(0, ANI_MAX)
    axN.set_ylim(bottom=0)
    axN.set_ylabel('$n$ (m$^{-3}$)')
    axN.set_xlabel('$l_{1d}/l_{2d}$')


def collect_details(index):
    """ Download the data needed to reproduce the computation from the header. """

    # Assume they all exist and if not then the code after will crash.
    path = get_path(index)
    with open(path) as f:
        for line in f:
            if 'N = ' in line:
                N = int(line.split()[3])
            if 'err = ' in line:
                err = float(line.split()[3])
            if 'outer square =' in line:
                line = line.replace(']', '')
                Lp = float(line.split()[-1])
            if 'B = ' in line:
                B = float(line.split()[3])

    return N, err, Lp, B


def show_fs_situation(index, ani, *, ax='new', units='SI'):
    """ Use the info in the file to show the mean free path distribution. """

    N, err, Lp, B = collect_details(index)
    if ax == 'new':
        f, ax = plt.subplots()
        f.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.92)

    LsetO = np.array([3, Lp * ani, Lp])
    LsetI = np.array([0, Lp])
    LsetC = np.array([3, Lp * ani, Lp])

    xxb, ccc = code.smooth_fs_2d('outer_square')
    code.show_smooth_fs_2d('outer_square', xxb, ccc,
                           LsetO, size_factor=1e17, ax=ax)
    f1, ax1 = code.show_smooth_Lorb_2d('outer_square', xxb, ccc, LsetO)

    xxb, ccc = code.smooth_fs_2d('inner_square')
    code.show_smooth_fs_2d('inner_square', xxb, ccc,
                           LsetI, size_factor=1e17, ax=ax)
    f2, ax2 = code.show_smooth_Lorb_2d('inner_square', xxb, ccc, LsetI)

    xxb, ccc = code.smooth_fs_1d('chain')
    code.show_smooth_fs_1d('chain', xxb, ccc, LsetC, size_factor=1e17, ax=ax)
    f3, ax3 = code.show_smooth_Lorb_1d('chain', xxb, ccc, LsetC)

    ax.plot([-np.pi / code.A] * 2, [-np.pi / code.B,
                                    np.pi / code.B], color='black', lw=2)
    ax.plot([np.pi / code.A] * 2, [-np.pi / code.B,
                                   np.pi / code.B], color='black', lw=2)
    ax.plot([-np.pi / code.A, np.pi / code.A],
            [-np.pi / code.B] * 2, color='black', lw=2)
    ax.plot([-np.pi / code.A, np.pi / code.A],
            [np.pi / code.B] * 2, color='black', lw=2)

    ax.set_aspect('equal')
    ax.set_xlim(-np.pi / code.A, np.pi / code.A)
    ax.set_ylim(-np.pi / code.B, np.pi / code.B)

    ax.set_xlabel('$k_x$ (1/m)')
    ax.set_ylabel('$k_y$ (1/m)')
    if units != 'SI':
        ax.set_xlabel('$k_xa$')
        ax.set_ylabel('$k_yb$')
        ax.set_xticks([-np.pi / code.A, 0, np.pi / code.A])
        ax.set_yticks([-np.pi / code.B, 0, np.pi / code.B])
        ax.set_xticklabels(['-$\\pi$', '0', '$\\pi$'])
        ax.set_yticklabels(['-$\\pi$', '0', '$\\pi$'])

    f.savefig('05 recent/_last_fs_with_L.png', dpi=400)
    f1.suptitle(f'$l$ orbit inner square Lc/Lp={ani:.1f}')
    f1.savefig('05 recent/_last_Lorb_inner_square.png', dpi=400)
    f2.suptitle(f'$l$ orbit outer square Lc/Lp={ani:.1f}')
    f2.savefig('05 recent/_last_Lorb_outer_square.png', dpi=400)
    f3.subplots_adjust(left=0.2, right=0.95)
    f3.suptitle(f'$l$ orbit chain Lc/Lp={ani:.1f}')
    f3.savefig('05 recent/_last_Lorb_chain_square.png', dpi=400)
    return f, ax


def show_combination(index):
    """ Show a multi-panel plot with lots of information. """

    aani, sss, eee = download(index)
    rrr, errr = full_conversion(sss, eee)
    _, _, _, constB = collect_details(index)

    if not os.path.isdir('05 recent'):
        os.mkdir('05 recent')
    for file in os.listdir('05 recent'):
        os.remove(os.path.join('05 recent', file))

    f, axes = plt.subplots(nrows=2, ncols=2,
                           figsize=(20, 13), num=f'file {index}')
    f.subplots_adjust(left=0.1, right=0.98, bottom=0.1,
                      top=0.97, wspace=0.5, hspace=0.3)
    show_all_sigmas(aani, sss, eee, ax=axes[0][0])
    show_resistivity(aani, rrr, errr, ax=axes[0][1])
    show_nH_ani(aani, rrr, errr, constB, axA=axes[1][0], axN=axes[1][1])

    axes[0][0].annotate(f'$B$ = {constB:.0f} T', [0.55, 0.52], ha='center',
                        va='center', xycoords='figure fraction')
    axes[0][1].tick_params(axis='x', pad=15)
    axes[1][0].tick_params(axis='x', pad=15)
    axes[1][1].tick_params(axis='x', pad=15)

    axes[0][0].annotate('a)', (0.03, 0.97),
                        xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('b)', (0.57, 0.97),
                        xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('c)', (0.03, 0.47),
                        xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('d)', (0.57, 0.47),
                        xycoords='figure fraction', weight='bold')

    f.savefig('05 recent/_last_combi.png', dpi=400)
    f2, _ = show_fs_situation(index, 0.3, units='normed')
    f2.savefig('05 recent/_last_situation.png', dpi=400)


# Cuts off the x axis even if calculations run further.
#   Above 3 is really not realistic given the Ando data and the fact
#   that these calculations are run at low T.
ANI_MAX = 3
plt.rc('font', size=28)
show_combination(1)
plt.show()
