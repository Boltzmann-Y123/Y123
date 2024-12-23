# Scipt to show results fresh off the press.
# Turned into a neat standardised view specific to the aims of the program:
#   How is nH in YBCO affected from the 2d plane carrier density by the 1d chains?
import os
import numpy as np
import matplotlib.pyplot as plt

# This unfortunately takes about 20 seconds
# because all functions with signature are compiled by numba.
code = __import__('04_compute')
planar = __import__('04_planar_density')


def get_path(index):
    """ Responsible for: Where is this data exactly.

    Raises if does not exist.
    """

    direc = '02 output Bsweep'
    if not os.path.isdir(direc):
        raise IOError('No data output directory.')

    path = os.path.join(direc, f'{index:04d}.dat')
    if not os.path.isfile(path):
        raise IOError(f'No data output file #{index:04d}')
    return path


def download(index):
    """ Get the magnetic fields, and for each the conductivities and errors. """

    path = get_path(index)
    BB = []
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
            BB.append(nums[0])
            sss.append(nums[1::2])
            eee.append(nums[2::2])
            assert(len(eee[-1]) == len(sss[-1]))
    BB = np.array(BB)

    wh = BB > MINI_B
    return BB[wh], np.array(sss)[wh], np.array(eee)[wh]


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
            if 'Lset' in line:
                part = line.split('[')[1]
                part = part.split(']')[0]
                part = part.replace(',', ' ')
                Lset = [float(x) for x in part.split()]

                if 'outer square' in line:
                    LsetO = np.array(Lset)
                elif 'inner square' in line:
                    LsetI = np.array(Lset)
                elif 'chain' in line:
                    LsetC = np.array(Lset)
                else:
                    raise IOError(f'Unknown band: {line}')

    return N, err, LsetO, LsetI, LsetC


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


def show_all_sigmas(BB, sss, eee, *, ax='new'):
    """ All the conductivity components. """

    sss = np.array(sss) / 1e8
    eee = np.array(eee) / 1e8

    ms = 6

    if ax == 'new':
        f, ax = plt.subplots(figsize=(10, 5))
        f.subplots_adjust(left=0.1, bottom=0.15, right=0.85, top=0.97)

    ax.errorbar(BB, sss[:, 0], yerr=eee[:, 0], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, O}$')
    ax.errorbar(BB, sss[:, 1], yerr=eee[:, 1], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, O}$')
    ax.errorbar(BB, sss[:, 2], yerr=eee[:, 2], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, O}$')

    ax.errorbar(BB, sss[:, 3], yerr=eee[:, 3], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, I}$')
    ax.errorbar(BB, sss[:, 4], yerr=eee[:, 4], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, I}$')
    ax.errorbar(BB, sss[:, 5], yerr=eee[:, 5], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, I}$')

    ax.errorbar(BB, sss[:, 6], yerr=eee[:, 6], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{aa, C}$')
    ax.errorbar(BB, sss[:, 7], yerr=eee[:, 7], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{ab, C}$')
    ax.errorbar(BB, sss[:, 8], yerr=eee[:, 8], lw=4, marker='o', ms=ms,
                elinewidth=2, label='$\u03C3_{bb, C}$')

    ax.set_xlim(0, max(BB))
    ax.set_xlabel('$B$ (T)')
    ax.set_ylabel('$\u03C3$ (1/\u03BC\u03A9cm)')
    ax.legend(frameon=False, bbox_to_anchor=[1, 1], loc='upper left')


def show_resistivity(BB, rrr, errr, *, ax='new'):
    """ In here to easily turn on/off and keep overview """

    if ax == 'new':
        f, ax = plt.subplots()

    ax.errorbar(BB, rrr[:, 0], yerr=errr[:, 0], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{aa}$')
    ax.errorbar(BB, rrr[:, 2], yerr=errr[:, 2], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{bb}$')
    ax.errorbar(BB, rrr[:, 1], yerr=errr[:, 1], lw=4, marker='o', ms=6,
                elinewidth=2, label='$\u03C1_{ba}$')
    ax.set_xlim(0, max(BB))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$B$ (T)')
    ax.set_ylabel('$\u03C1$ (\u03BC\u03A9cm)')
    ax.legend(frameon=False)


def show_nH_ani(BB, rrr, errr, Lc=0, Lp=0, *, axA='new', axN='new'):
    """ Show n, nH and nH/ani. The true result. """

    if axA == 'new':
        f, axA = plt.subplots()
    aani = rrr[:, 0] / rrr[:, 2]
    # Error propagation like in convert_rho()
    eaani = (1 / rrr[:, 2])**2 * errr[:, 0]**2
    eaani += (-rrr[:, 0] / rrr[:, 2]**2)**2 * errr[:, 2]**2
    eaani = np.sqrt(eaani)

    axA.errorbar(BB, aani, yerr=eaani, lw=4, color='tab:red', marker='o', ms=6)
    axA.scatter(BB, aani, s=50, color='tab:red')
    axA.set_xlim(0, max(BB))
    axA.set_ylim(0, max(aani) * 1.1)
    axA.set_xlabel('$B$ (T)')
    axA.set_ylabel('$\u03C1_{aa}/\u03C1_{bb}$')
    if max(aani) < 1.95 and min(aani) > 1.85:
        # Specifically for the isotropic case control for a nicer figure.
        axA.set_yticks([1.89, 1.9, 1.91, 1.92])
        axA.set_ylim(1.89, 1.925)

    if Lc > 0 and Lp > 0:
        axA.annotate('$l_{1d}$ = ' + f'{Lc * 1e9:.1f} nm', [0.55, 0.54],
                     ha='center', va='center', xycoords='figure fraction')
        axA.annotate('$l_{2d}$ = ' + f'{Lp * 1e9:.1f} nm', [0.55, 0.5],
                     ha='center', va='center', xycoords='figure fraction')

    if axN == 'new':
        f, axN = plt.subplots()
    nBZ = 1 / code.A / code.C / code.B
    n2d = planar.get_planar_carrier_density()
    # planar.create_larger_2d_fs(show=True)
    print(f'Correct (pre-hybridise) planar doping: p={n2d / nBZ / 2 - 1:.2f}')

    axN.plot([0, max(BB) * 2], [n2d] * 2, lw=2,
             dashes=[6, 2], color='black', label='n$_{pl}$')

    # Again error propagation
    RH = rrr[1:, 1] / BB[1:] * 10
    eRH = errr[1:, 1] / BB[1:] * 10
    nH = 1e9 / RH / code.Q
    enH = np.sqrt((-1e9 / RH**2 / code.Q)**2 * eRH**2)
    nHf = nH / aani[1:]
    enHf = (1 / aani[1:])**2 * enH**2
    enHf += (-nH / aani[1:]**2)**2 * eaani[1:]**2
    enHf = np.sqrt(enHf)

    axN.errorbar(BB[1:], nH, yerr=enH, color='.4', elinewidth=2,
                 label='$n_H$', lw=4, marker='o', ms=6)
    axN.errorbar(BB[1:], nHf, yerr=enHf, lw=4, marker='o', ms=6, elinewidth=2,
                 color='tab:blue', label="$n_H\u03C1_{bb}/\u03C1_{aa}$")
    axN.legend(frameon=False, loc='lower right')
    axN.set_xlim(0, max(BB))
    axN.set_ylim(bottom=0)
    axN.set_ylabel('$n$ (m$^{-3}$)')
    axN.set_xlabel('$B$ (T)')


def show_fs_situation(index, *, ax='new', units='SI'):
    """ Use the info in the file to show the mean free path distribution. """

    N, err, LsetO, LsetI, LsetC = collect_details(index)
    if ax == 'new':
        f, ax = plt.subplots()
        f.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.92)

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

    ax.set_xlabel('$k_x$ (1/m)')
    ax.set_ylabel('$k_y$ (1/m)')

    ax.plot([-np.pi / code.A] * 2, 
            [-np.pi / code.B, np.pi / code.B], color='black', lw=2)
    ax.plot([np.pi / code.A] * 2, 
            [-np.pi / code.B, np.pi / code.B], color='black', lw=2)
    ax.plot([-np.pi / code.A, np.pi / code.A],
            [-np.pi / code.B] * 2, color='black', lw=2)
    ax.plot([-np.pi / code.A, np.pi / code.A],
            [np.pi / code.B] * 2, color='black', lw=2)

    ax.set_aspect('equal')
    ax.set_xlim(-np.pi / code.A, np.pi / code.A)
    ax.set_ylim(-np.pi / code.B, np.pi / code.B)
    if units != 'SI':
        ax.set_xlabel('$k_xa$')
        ax.set_ylabel('$k_yb$')
        ax.set_xticks([-np.pi / code.A, 0, np.pi / code.A])
        ax.set_yticks([-np.pi / code.B, 0, np.pi / code.B])
        ax.set_xticklabels(['-$\\pi$', '0', '$\\pi$'])
        ax.set_yticklabels(['-$\\pi$', '0', '$\\pi$'])

    f.savefig('05 recent/_last_situation.png', dpi=400)
    f1.savefig('05 recent/_last_Lorb_inner_square.png', dpi=400)
    f2.savefig('05 recent/_last_Lorb_outer_square.png', dpi=400)
    f3.subplots_adjust(left=0.2, right=0.95)
    f3.savefig('05 recent/_last_Lorb_chain_square.png', dpi=400)
    return f, ax


def show_multipanel(index):
    """ Show a multi-panel plot with comprehensive information about this B sweep. """

    BB, sss, eee = download(index)
    rrr, errr = full_conversion(sss, eee)
    N, err, LsetO, LsetI, LsetC = collect_details(index)

    if not os.path.isdir('05 recent'):
        os.mkdir('05 recent')
    for file in os.listdir('05 recent'):
        os.remove(os.path.join('05 recent', file))

    f, axes = plt.subplots(nrows=2, ncols=2, 
                           figsize=(20, 15), num=f'file {index}')
    f.subplots_adjust(left=0.1, right=0.96, bottom=0.1,
                      top=0.97, wspace=0.6, hspace=0.3)
    show_all_sigmas(BB, sss, eee, ax=axes[0][0])
    show_resistivity(BB, rrr, errr, ax=axes[0][1])

    Lc = LsetC[1] if LsetC[0] in [0, 3] else 0
    Lp = LsetO[2] if LsetO[0] == 3 else 0
    Lp = LsetO[1] if LsetO[0] == 0 and Lp == 0 else Lp
    show_nH_ani(BB, rrr, errr, Lc, Lp, axA=axes[1][0], axN=axes[1][1])

    axes[0][0].annotate('a)', (0.03, 0.97), xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('b)', (0.55, 0.97), xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('c)', (0.03, 0.45), xycoords='figure fraction', weight='bold')
    axes[0][0].annotate('d)', (0.55, 0.45), xycoords='figure fraction', weight='bold')
    f.savefig('05 recent/_last_combi.png', dpi=400)
    show_fs_situation(index, units='normed')


# Files:
# 1: Lc=Lp=7.5 nm (N=5001)
# 2: Lc=2.5 nm, Lp=7.5 nm (N=5001)
# 3: Lc=15 nm, Lp=7.5 nm (N=5001)
# 4: Lc=3.75 nm, Lp=7.5 nm (N=5001)

# The very low B regime with the very small wc*tau from modelling 50 K
#   data requires some insane precision to get sigma_xy proper with a Fermi surface
#   that only marginally adheres to symmetry principles.
MINI_B = 1
MAXI_B = 100
plt.rc('font', size=28)
show_multipanel(1)

plt.show()
