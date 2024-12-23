# Initiates computations
# Scripting 04_compute essentially, all the dirty adjustments happen
# here as much as possible to get freedom and robustness separated.
# Author: Roemer Hinlopen

import datetime
import multiprocessing as mp
import os
import time
import numpy as np
import matplotlib.pyplot as plt


code = __import__('04_compute')


def dummy_show_L(LsetO, LsetI, LsetC):
    """ Just a playground to see the L variation before committing. """

    LsetO = np.array(LsetO)
    LsetI = np.array(LsetI)
    LsetC = np.array(LsetC)

    xxb1, ccc1 = code.smooth_fs_2d('outer_square')
    xxb2, ccc2 = code.smooth_fs_2d('inner_square')
    pphi = np.linspace(0, 2 * np.pi, 1000)
    LL1 = []
    LL2 = []
    for phi in pphi:
        L1 = code.get_L_2d(phi, LsetO, xxb1, ccc1)
        LL1.append(L1)
        L2 = code.get_L_2d(phi, LsetI, xxb2, ccc2)
        LL2.append(L2)
    plt.figure()
    plt.plot(pphi, LL1, label='Outer square')
    plt.plot(pphi, LL2, label='Inner square')
    plt.legend()
    plt.xlabel('$\\phi$ (rad)')
    plt.ylabel('$L$ (m)')

    kkx = np.linspace(-np.pi / code.A, np.pi / code.A, 1000)
    xxb3, ccc3 = code.smooth_fs_1d('chain')
    LL3 = []
    for kx in kkx:
        L3 = code.get_L_1d(kx, LsetC, xxb3, ccc3)
        LL3.append(L3)
    plt.figure()
    plt.plot(kkx, LL3, label='Chain')
    plt.legend()
    plt.xlabel('$k_x$ (1/m)')
    plt.ylabel('$L$ (m)')

    # Show in 2d with 'arrows'
    f, ax = plt.subplots()
    xxb, ccc = code.smooth_fs_2d('outer_square')
    code.show_smooth_fs_2d('outer_square', xxb, ccc, LsetO, size_factor=1e17, ax=ax)
    xxb, ccc = code.smooth_fs_2d('inner_square')
    code.show_smooth_fs_2d('inner_square', xxb, ccc, LsetI, size_factor=1e17, ax=ax)
    xxb, ccc = code.smooth_fs_1d('chain')
    code.show_smooth_fs_1d('chain', xxb, ccc, LsetC, size_factor=1e17, ax=ax)

    print(flush=True)
    plt.show()


def execute(B):
    """ Main entrypoint for multiprocessing """

    st = time.process_time()
    ssO, eeO = code.sigma_auto(B, N, np.array(LsetO), 'outer_square', err, loud=LOUD)
    ssI, eeI = code.sigma_auto(B, N, np.array(LsetI), 'inner_square', err, loud=LOUD)
    ssC, eeC = code.sigma_auto(B, N, np.array(LsetC), 'chain', err, loud=LOUD)
    timer = time.process_time() - st
    print(f'> Finished all 3 bands at B={B} T in {timer:.2f} s', flush=True)
    return [[ssO, eeO], [ssI, eeI], [ssC, eeC]]


def store_results(BB, rrr):
    """ Includes settings in the header of the file. """

    direc = '02 output Bsweep'
    if not os.path.isdir(direc):
        os.mkdir(direc)

    index = 1
    while os.path.isfile(os.path.join(direc, f'{index:04d}.dat')):
        index += 1
        assert(index < 10000)

    rrr = np.array(rrr)
    assert(len(np.shape(rrr)) == 4)
    assert(np.shape(rrr)[3] == 3)  # xx, xy, yy
    assert(np.shape(rrr)[2] == 2)  # sigma, error
    assert(np.shape(rrr)[1] == 3)  # nr of bands
    assert(np.shape(rrr)[0] == len(BB))  # nr of magnetic fields

    path = os.path.join(direc, f'{index:04d}.dat')
    with open(path, 'w') as f:
        f.write(f'# Generated {datetime.datetime.now()}\n')
        f.write(f'# Execution time {time.time() - ST:.1f} s\n')
        f.write(f'# N = {N} (number of FS points the same for each FS) \n')
        f.write(f'# err = {err} (integral error aim the same for each FS) \n')
        f.write(f'# Lset mean free path setting outer square = {LsetO}\n')
        f.write(f'# Lset mean free path setting inner square = {LsetI}\n')
        f.write(f'# Lset mean free path setting chain = {LsetC}\n')
        f.write(f'# Units of B are Tesla, sigma and error is 1/(Ohm meter)\n')
        f.write(f'# O = outer square, I = inner square, C = chain\n\n')

        f.write('B sxxO exxO sxyO exyO syyO eyyO '
                'sxxI exxI sxyI exyI syyI eyyI '
                'sxxC exxC sxyC exyC syyC eyyC\n')

        for B, data in zip(BB, rrr):
            line = f'{B:.10f} {data[0][0][0]:15.5f} {data[0][1][0]:15.5f} '
            line += f'{data[0][0][1]:15.5f} {data[0][1][1]:15.5f} '
            line += f'{data[0][0][2]:15.5f} {data[0][1][2]:15.5f} '

            line += f'{data[1][0][0]:15.5f} {data[1][1][0]:15.5f} '
            line += f'{data[1][0][1]:15.5f} {data[1][1][1]:15.5f} '
            line += f'{data[1][0][2]:15.5f} {data[1][1][2]:15.5f} '

            line += f'{data[2][0][0]:15.5f} {data[2][1][0]:15.5f} '
            line += f'{data[2][0][1]:15.5f} {data[2][1][1]:15.5f} '
            line += f'{data[2][0][2]:15.5f} {data[2][1][2]:15.5f}\n'
            f.write(line)

    print(f'> Saved results into file {index:04d}.dat')


# Settings
N = 5001
err = 1e-6

# Isotropic // data 0001:
LsetO = [0, 7.5e-9]
LsetI = [0, 7.5e-9]
LsetC = [0, 7.5e-9]

# Chain and plane different // data 0002
# Lplane = 7.5e-9
# Lchain = 2.5e-9
# LsetO = [3, Lchain, Lplane]
# LsetI = [0, Lplane]
# LsetC = [3, Lchain, Lplane]

# Chain and plane different  // data 0003
# Lplane = 7.5e-9
# Lchain = 15e-9
# LsetO = [3, Lchain, Lplane]
# LsetI = [0, Lplane]
# LsetC = [3, Lchain, Lplane]

# Chain and plane different  // data 0004
# Lplane = 7.5e-9
# Lchain = 3.75e-9
# LsetO = [3, Lchain, Lplane]
# LsetI = [0, Lplane]
# LsetC = [3, Lchain, Lplane]

ST = time.time()
LOUD = True

if __name__ == '__main__':

    dummy_show_L(LsetO, LsetI, LsetC)
    BB = [1, 2, 3, 4, 5, 7.5] + list(np.arange(10, 102, 5))
    BB += list(np.arange(125, 500, 25))
    BB += list(np.arange(500, 5010, 50))

    with mp.Pool(4) as pool:
        rrr = pool.map(execute, BB)
    store_results(BB, rrr)
