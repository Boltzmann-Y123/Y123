#   Tests symmetries which must be present
#   Tests sign and size of the Hall effect
#   Tests conductivity against Drude theory
#   Tests conductivity against impeded cyclotron motion (Hinlopen2022)
#   Tests the errors produced are sensible
# All should pass.
# (A number of tests just are slow)
# Author: Roemer Hinlopen

import unittest
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import time

code = __import__('04_compute')


def timer(func):
    def replacement(*a, **kw):
        st = time.process_time()
        r = func(*a, **kw)
        timer = time.process_time() - st
        if timer > 5:
            print(f'\n   > Test {func.__name__} is slow at {timer:.1f} s')
        return r
    return replacement


class TestSmoothing(unittest.TestCase):

    @timer
    def test_smooth_and_stabilise(self):
        """ Made during development. """

        # Basically, the plots verify this result and this
        # was developed as the first iteration of the smoothing.
        # and this is here embedded as a test to ensure no changes
        # are made to the code that change this result.
        #
        # More detailed tests surely could be made, but the result
        # really speaks for itself here.
        xx = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        yy = np.sin(xx)

        xx += np.random.random(len(xx)) * 0.2 * (xx[1] - xx[0])
        yy += np.random.random(len(yy)) * 0.05

        # plt.figure('Raw')
        # plt.scatter(xx, yy)
        yy2 = code.smooth(xx, yy, 9, 1, 2 * np.pi)
        # plt.plot(xx, yy2)

        xfeed = list(xx - 2 * np.pi) + list(xx) + list(xx + 2 * np.pi)
        yy3 = list(yy2) * 3
        Interp = si.CubicSpline(xfeed, yy3)

        xplt = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        # plt.plot(xplt, np.sin(xplt))
        # plt.plot(xplt, Interp(xplt))

        # plt.figure('Derivative')
        delta = 2 * np.pi * 1e-4
        dyy = (Interp(xplt + delta) - Interp(xplt)) / delta
        # plt.plot(xplt, dyy)
        # plt.plot(xplt, np.cos(xplt))
        stab_dyy = code.smooth(xplt, dyy, 61, 1, 2 * np.pi)
        stab_dyy += np.mean(dyy) - np.mean(stab_dyy)
        # plt.plot(xplt, stab_dyy, color='tab:red')

        stabilised_xx = xplt[::2]
        stabilised_yy = np.zeros(len(stab_dyy) // 2)

        for i in range(1, len(stabilised_xx)):
            delta = stab_dyy[2 * i]
            delta += 2 * stab_dyy[(2 * i + 1) % len(stab_dyy)]
            delta += stab_dyy[(2 * i + 2) % len(stab_dyy)]
            delta *= 0.5 * (xplt[1] - xplt[0])
            stabilised_yy[i] = stabilised_yy[i - 1] + delta
        stabilised_yy += np.mean(Interp(xplt)) - np.mean(stabilised_yy)

        # plt.figure('Stabilised')
        # plt.plot(stabilised_xx, stabilised_yy, color='tab:red')
        # plt.plot(xplt, np.sin(xplt))

        # The direct algorithm should be indistinguishable
        xxs, yys = code.stabilised_deriv(xx, yy2, 1000, 61, 2 * np.pi)
        for x, y in zip(xxs, yys):
            i = np.argmin(np.abs(stabilised_xx - x))
            assert(abs(y - stabilised_yy[i]) < 1e-6)

        # plt.figure('Stabilised')
        # plt.plot(xxs, yys, color='tab:purple')

    @timer
    def test_index_below(self):
        """ There are a few corner cases here that might rarely occur in isolation
        but are important for a full sigma calculation. """

        # So the idea is to get the index of the x value just below what is given.
        # But that leaves corner cases:
        xx = np.array([-2, 1, 3, 300, 3e6])
        self.assertEqual(code._binary_index_below(-1, xx), 0)

        # Exact lower limit is a corner case.
        self.assertEqual(code._binary_index_below(-2, xx), 0)

        # Above upper limit is no problem, below lower limit is.
        self.assertEqual(code._binary_index_below(4e6, xx), 4)
        self.assertRaises(AssertionError, code._binary_index_below, -3, xx)

        # I don't actually care about the edges, because
        #   splines and their derivatives are continuous,
        # but good to just have consistent behaviour regardless.
        self.assertEqual(code._binary_index_below(1, xx), 1)

        # No tests for unsorted xxbreak or NaNs, that is
        # undefined behaviour.

    @timer
    def test_eval_ppoly(self):
        """ Quick test """

        xx = np.array([0, 1, 2, 3, 4., 5, 6])
        ccc = np.array([[0, 1], [1., 1], [2, 1], [
                       3, 1], [4, 1], [5, 1], [6, 1]]).T

        # 2x+1
        v = code.eval_ppoly(9.5, xx, ccc, 7.)
        self.assertAlmostEqual(v, 2 * (2.5 - 2) + 1)

        dv = code.eval_dppoly(9.5, xx, ccc, 7.)
        self.assertAlmostEqual(dv, 2)

    @timer
    def test_orthogonality_L(self):
        """ Especially in 1d this was a problem for a bit. """

        xxb, ccc = code.smooth_fs_2d('inner_square')
        dkydkx = code.eval_dppoly(1e9, xxb, ccc, np.pi / code.A)
        parallel = np.array([1, dkydkx])
        parallel /= np.linalg.norm(parallel)
        Ldir = code.get_Ldir_1d(1e9, xxb, ccc)

        # Must be properly orthogonal
        self.assertAlmostEqual(np.dot(parallel, Ldir), 0, delta=1e-10)

    @timer
    def test_show_smoothed_fs(self):
        """ The goal of all this: Generate a continuous Fermi surface with
        smooth Fermi velocity. """

        print('> Skipping showing the FSs')
        return

        xxb, ccc = code.smooth_fs_2d('inner_square')
        code.show_smooth_fs_2d('inner_square', xxb, ccc, np.array([0, 1e-8]))

        xxb, ccc = code.smooth_fs_2d('outer_square')
        code.show_smooth_fs_2d('outer_square', xxb, ccc, np.array([0, 1e-8]))

        xxb, ccc = code.smooth_fs_1d()
        code.show_smooth_fs_1d(xxb, ccc, np.array([0, 1e-8]))

        # plt.show()

    @timer
    def test_density_2d(self):

        # Flat radius 1e9 so result is easy:
        xxb, ccc = code.smooth_fs_2d('iso_2d')
        expect = 1e9**2 / (2 * np.pi * code.C)
        n = code.compute_density_2d(xxb, ccc)
        self.assertAlmostEqual(expect, n, delta=expect * 1e-5)

        # Remembered result.
        # Changed by 1/1000 (7.8e24) when introducing mirror symmetry.
        xxb, ccc = code.smooth_fs_2d('outer_square')
        n = code.compute_density_2d(xxb, ccc)
        self.assertAlmostEqual(n, 7.698907e27, delta=1e21)

    @timer
    def test_mirror_symmetry(self):
        print(' To do mirror symm test')


class TestCyclotronMotion(unittest.TestCase):

    @timer
    def test_pathlength(self):
        """ A nasty but simple test: The circumference of a circle.
        That includes points at phi=0, pi which are retrograde in the (x,y) plane,
        half the circle is retrograde and the answer is known analytically.

        This is solved by a basis transformation in the code, always aligning
        the first and third point with a new x axis.
        (Yet this passed first try. Somewhat proud)
        """

        # First point = last point.
        pphi = np.linspace(0, 2 * np.pi, 121)
        xx = np.cos(pphi)
        yy = np.sin(pphi)

        L = 0
        for i in range(0, len(xx) - 2, 2):
            dL = code.fit_and_measure_pathlength(xx[i], xx[i + 1], xx[i + 2],
                                                 yy[i], yy[i + 1], yy[i + 2])
            L += dL

        # Believe it or not, this was right the first time.
        # Including retrograde. ADMR experience.
        assert(abs(L - 2 * np.pi) < 1e-6)

    @timer
    def test_performance_pathlength(self):
        """ So pathlengths can be determined by subdividing into more and more
        straight line segments, or by dividing into more and more parabola.
        The latter has better complexity but worse overhead.
        The more expensive Fermi surface points are, the better the parabola.
        This code was used to test correctness, but more importantly to determine
        that for this particular case the cutoff is only at 1e-10 accuracy,
        which means the linear segments simple algorith outperforms parabolas.
        """

        # Euclidean.
        # ---------
        # This is raw output made with a global variable to track
        # the number of Fermi surface points evaluated beyond the
        # starting two for various precisions. Settings:
        #       get_pathlength_euc_2d(1, 2, xxb1, ccc1, err)
        #
        #  > count: 1 | 1e-1
        # 6125339655.336141
        #  > count: 1 | 1e-2
        # 6125339655.336141
        #  > count: 1 | 1e-3
        # 6125339655.336141
        #  > count: 19 | 1e-4
        # 6131407418.360569
        #  > count: 47 | 1e-5
        # 6131438876.431582
        #  > count: 149 | 1e-6
        # 6131452485.140648
        #  > count: 493 | 1e-7
        # 6131453380.455896
        #  > count: 1491 | 1e-8
        # 6131453534.247116

        # Parabolic
        # ---------
        # The same procedure, the same pathlength:
        #  > count 2 | 0.1
        # 6130185002.272682
        #  > count 2 | 0.01
        # 6130185002.272682
        #  > count 2 | 0.001
        # 6130185002.272682
        #  > count 14 | 0.0001
        # 6131412147.348831
        #  > count 26 | 1e-05
        # 6131450258.451143
        #  > count 66 | 1e-06
        # 6131453478.870838
        #  > count 134 | 1e-07
        # 6131453549.276047
        #  > count 262 | 1e-08
        # 6131453560.107965
        #  > count 466 | 1e-09
        # 6131453560.033312
        #  > count 858 | 1e-10
        # 6131453560.138586
        #  > count 1442 | 1e-11
        # 6131453560.155243
        #
        # >> After mirror symmetry the length changes
        #   by 0.5 %. The new answer at 1e-13 error is 6119617448.569095

        answer = 6119617448.569095
        xxb1, ccc1 = code.smooth_fs_2d('inner_square')
        st0 = time.process_time()
        for err, reps in zip([1e-1, 1e-3, 1e-6, 1e-11], [10000, 3000, 300, 10]):
            v = code.get_pathlength_euc_2d(1, 2, xxb1, ccc1, err)
            assert(abs(v - answer) < answer * err)
            st = time.process_time_ns()
            for _ in range(reps):
                code.get_pathlength_euc_2d(1, 2, xxb1, ccc1, err)
                code.get_pathlength_euc_2d(2, 3, xxb1, ccc1, err)
                code.get_pathlength_euc_2d(3, 4, xxb1, ccc1, err)
                code.get_pathlength_euc_2d(4, 5, xxb1, ccc1, err)
            timer1 = 1e-9 * (time.process_time_ns() - st) / reps / 4

            v = code.get_pathlength_par_2d(1, 2, xxb1, ccc1, err)
            assert(abs(v - answer) < answer * err)
            st = time.process_time_ns()
            for _ in range(reps):
                code.get_pathlength_par_2d(1, 2, xxb1, ccc1, err)
                code.get_pathlength_par_2d(2, 3, xxb1, ccc1, err)
                code.get_pathlength_par_2d(3, 4, xxb1, ccc1, err)
                code.get_pathlength_par_2d(4, 5, xxb1, ccc1, err)
            timer2 = 1e-9 * (time.process_time_ns() - st) / reps / 4
            print(f'Pathlength benchmark err={err}: Euclidean {timer1:.2e} s, '
                  f'parabolic {timer2:.2e} s')
        t = time.process_time() - st0
        print('Conclusion: Use Euclidean. '
              f'FS points are too cheap. ({t:.1f} s)')

    @timer
    def test_survival(self):
        """ Test survival over a full orbit to a known result """

        # A trusted hard-coded value.
        B = 30
        Lset = np.array([1, 500e-9, 6.5e9])
        loop = code.show_survival_2d('outer_square', B, Lset, show=False)
        self.assertAlmostEqual(loop, 0.0622, delta=0.0001)

        # An intuitive result that matches this is order of magnitude
        xxb, ccc = code.smooth_fs_2d('outer_square')
        pphi = np.linspace(0, 2 * np.pi, 1000)
        dists = [code.get_pathlength_2d(p1, p2, xxb, ccc, 1e-6)
                 for p1, p2 in zip(pphi[:-1], pphi[1:])]
        circumference = sum(dists)

        # This Lset is optimised such that really you get the 500e-9 m.
        # Albeit with substantial anisotropy which makes the real result
        # less than 11% as expected, but not 1% (or 90%).
        Lvals = [code.get_L_2d(p, Lset, xxb, ccc) for p in pphi]
        avg_L = np.mean(Lvals)
        self.assertAlmostEqual(avg_L, 5e-7, delta=1e-8)
        expect_dist = code.Q / code.HBAR * B * avg_L
        expect_odds = np.exp(-circumference / expect_dist)
        self.assertAlmostEqual(expect_odds, 0.11, delta=0.01)

    @timer
    def test_survival_1d(self):
        """ Separate function, separate test. """

        # A comparable result, a hard-coded value to fall back on.
        B = 30
        Lset = np.array([0, 500e-9])
        loop = code.show_survival_1d('chain', B, Lset, show=False)
        exp_wct = code.Q * B * Lset[1] / code.HBAR
        exp_wct /= 2 * np.pi / code.A
        exp_prob = np.exp(-1 / exp_wct)
        self.assertAlmostEqual(exp_prob, 0.486, delta=0.001)
        self.assertAlmostEqual(loop, 0.4805, delta=0.001)

        xxb, ccc = code.smooth_fs_1d('chain')
        kkx = np.linspace(-np.pi / code.A, np.pi / code.A, 100)
        dists = [code.get_pathlength_1d(kx1, kx2, xxb, ccc, 1e-6)
                 for kx1, kx2 in zip(kkx[:-1], kkx[1:])]
        length = sum(dists)
        self.assertAlmostEqual(length, 2 * np.pi / code.A,
                               delta=0.05 * np.pi / code.A)

    @timer
    def test_cyclotron_integration_partial_Lvec(self):
        """ This is depth-first recursion for time-ordered integration. It is written with
        few branches, but quite complex if not handled and understood properly. """

        B = 1e-6
        Lset = np.array([1, 500e-9, 6.5e9])
        xxb, ccc = code.smooth_fs_2d('outer_square')
        pphi = np.linspace(0, 2 * np.pi, 500)

        st = time.process_time_ns()
        data = np.array([code.partial_Lvec_2d(p1, p2, B, Lset, xxb, ccc, 1e-6)
                         for p1, p2 in zip(pphi[:-1], pphi[1:])])
        timer = (time.process_time_ns() - st) / 1e9 / len(data)
        print(
            f'Benchmark partial Lvec (B=1e-6 T) to 1e-6 accuracy in {timer:.2e} s')

        # Given the low magnetic field value (wctau is about 2e-5 at B=0.0003 T)
        #   I expect the odds fall off to virtual 0 and the partial L's
        #   are the full L's.
        Lvals = [code.get_L_2d(phi, Lset, xxb, ccc) for phi in pphi]
        Lvecs = np.array([code.get_Ldir_2d(phi, xxb, ccc) * L
                          for phi, L in zip(pphi, Lvals)])

        # This is quite complex and convergence is a strong statement:
        # plt.figure()
        # plt.plot(pphi, Lvecs[:, 0])
        # plt.plot(pphi[:-1], data[:, 1])
        # plt.plot(pphi, Lvecs[:, 1])
        # plt.plot(pphi[:-1], data[:, 2])

        errs1 = []
        errs2 = []
        for point, Lvec in zip(data, Lvecs):
            Lmax = max(np.abs(Lvec))
            errs1.append(abs(point[1] - Lvec[0]) / Lmax)
            errs2.append(abs(point[2] - Lvec[1]) / Lmax)

        self.assertLess(max(errs1), 3 * 1e-6)
        self.assertLess(max(errs2), 3 * 1e-6)
        self.assertLess(max(data[:, 0]), 1e-50)

    @timer
    def test_cyclotron_integration_partial_Lvec_1d(self):
        """ Same test in 1d """

        B = 1e-6
        Lset = np.array([0, 500e-9])
        xxb, ccc = code.smooth_fs_1d('chain')
        kkx = np.linspace(0, 2 * np.pi / code.A, 100)

        data = np.array([code.partial_Lvec_1d(kx1, kx2, B, Lset, xxb, ccc, 3e-8)
                         for kx1, kx2 in zip(kkx[:-1], kkx[1:])])

        Lvals = [code.get_L_1d(kx, Lset, xxb, ccc) for kx in kkx]
        Lvecs = np.array([code.get_Ldir_1d(kx, xxb, ccc) * L
                          for kx, L in zip(kkx, Lvals)])

        # This is quite complex and convergence is a strong statement:
        # plt.figure()
        # plt.plot(pphi, Lvecs[:, 0])
        # plt.plot(pphi[:-1], data[:, 1])
        # plt.plot(pphi, Lvecs[:, 1])
        # plt.plot(pphi[:-1], data[:, 2])

        errs1 = []
        errs2 = []
        for point, Lvec in zip(data, Lvecs):
            Lmax = max(np.abs(Lvec))
            errs1.append(abs(point[1] - Lvec[0]) / Lmax)
            errs2.append(abs(point[2] - Lvec[1]) / Lmax)

        # This one is more stringent (and slower) than the
        # 2d test because the x component is small, typically
        # 1/200 times the size of the y component. To test that
        # rigorously too I need this accuracy.
        self.assertLess(max(errs1), 1e-6)
        self.assertLess(max(errs2), 3 * 1e-7)
        self.assertLess(max(data[:, 0]), 1e-50)

    @timer
    def test_cyclotron_odds(self):
        """ Test the result of partialL code agains
        the simpler and more trusted survival(). """

        # At B=30 one has about wctau=2.25 averaged
        # (6.2 % for an orbit, 90+ % per interval)
        B = 30
        Lset = np.array([1, 500e-9, 6.5e9])
        xxb, ccc = code.smooth_fs_2d('outer_square')
        pphi = np.linspace(0, 2 * np.pi, 50)

        st = time.process_time_ns()
        data = [code.partial_Lvec_2d(p1, p2, B, Lset, xxb, ccc, 1e-6)
                for p1, p2 in zip(pphi[:-1], pphi[1:])]
        timer = (time.process_time_ns() - st) / 1e9 / len(data)
        print(
            f'Benchmark partial Lvec (wctau=2.3) to 1e-6 accuracy in {timer:.2e} s')
        data = np.array(data)

        # Much simpler and above vetted code.
        # Not as much control over the recursion,
        # no integral being performed. Just odds. Simpler, trustworthier.
        odds = [code.survive_2d(p1, p2, B, Lset, xxb, ccc, 1e-10)
                for p1, p2 in zip(pphi[:-1], pphi[1:])]
        for o1, o2 in zip(odds, data[:, 0]):
            self.assertAlmostEqual(o1, o2, delta=1e-5)

        # The odds are *more* accurate than asked for.
        # This must naturally be the case because data has
        # the integral over time of the probability,
        # which is an addition of many probability errors.
        # But then there are also 50 errorss in this product.
        # Yet it turns out the resulting probability error is really small.
        loop = np.prod(odds)
        loop2 = np.prod(data[:, 0])
        self.assertAlmostEqual(loop, loop2, delta=1e-7)

    @timer
    def test_convolution(self):
        """ from L contribution over each section of the fs, to the full contribution """

        B = 30
        Lset = np.array([0, 500e-9])
        xxb, ccc = code.smooth_fs_2d('outer_square')
        pphi = np.linspace(0, 2 * np.pi, 100)
        data = [code.partial_Lvec_2d(p1, p2, B, Lset, xxb, ccc, 1e-6)
                for p1, p2 in zip(pphi[:-1], pphi[1:])]
        data = np.array(data)

        Lxx, Lyy = code.convolve_partial_Lvecs(
            data[:, 0], data[:, 1], data[:, 2])

        # About 49 % to make a full track
        loop = np.prod(data[:, 0])
        self.assertAlmostEqual(loop, 0.11, delta=0.01)
        wctau = abs(2 * np.pi / np.log(loop))
        self.assertAlmostEqual(wctau, 3, delta=0.1)

        # I cannot predict the analytical values, but I can explicitly
        # check the last value that was constructed with the convolultion from the
        # data to increase my trust here.
        # This is exact bar floating point precision is different on compiled code
        # than native Python, but really that precision is basically infinite.
        Lx1 = 0
        Ly1 = 0
        survival = 1
        for i in range(1, len(pphi) * 50):
            i = i % (len(pphi) - 1)
            Lx1 += survival * data[i][1]
            Ly1 += survival * data[i][2]
            survival *= data[i][0]
        self.assertLess(survival, 1e-10)
        self.assertGreater(survival, 0)
        self.assertAlmostEqual(Lx1, Lxx[1], delta=1e-15)
        self.assertAlmostEqual(Ly1, Lyy[1], delta=1e-15)

    @timer
    def test_convolution_1d(self):
        """ from L contribution over each section of the fs, to the full contribution """

        # Naive exp(-2pi/A / (q/hbar*LB)) gives me 49 % chance to live.
        #   The computations say 48 %, which is likely the slightly longer
        #   pathlength from the corrugation. Fine.
        B = 30
        Lset = np.array([0, 500e-9])
        xxb, ccc = code.smooth_fs_1d('chain')
        kkx = np.linspace(0, 2 * np.pi / code.A, 100)
        data = [code.partial_Lvec_1d(kx1, kx2, B, Lset, xxb, ccc, 1e-7)
                for kx1, kx2 in zip(kkx[:-1], kkx[1:])]
        data = np.array(data)

        Lxx, Lyy = code.convolve_partial_Lvecs(
            data[:, 0], data[:, 1], data[:, 2])

        # About 11 % to make a full loop, wctau=
        loop = np.prod(data[:, 0])
        self.assertAlmostEqual(loop, 0.48, delta=0.01)
        wctau = abs(2 * np.pi / np.log(loop))
        self.assertAlmostEqual(wctau, 8.5, delta=0.1)

        # Precise till floating point.
        # For the last value to be made in the convolution, so sensitive to errors.
        Lx1 = 0
        Ly1 = 0
        survival = 1
        for i in range(1, len(kkx) * 70):
            i = i % (len(kkx) - 1)
            Lx1 += survival * data[i][1]
            Ly1 += survival * data[i][2]
            survival *= data[i][0]
        self.assertLess(survival, 1e-10)
        self.assertGreater(survival, 0)
        self.assertAlmostEqual(Lx1, Lxx[1], delta=1e-15)
        self.assertAlmostEqual(Ly1, Lyy[1], delta=1e-15)


class TestSigma(unittest.TestCase):
    """ Test against Drude theory and a known case of Boltzmann transport theory to validate
    the final result is holistically accurate. """

    @timer
    def test_drude_2d(self):
        """ Test zero-field, but also zero-MR and Hall effect. """

        xxb, ccc = code.smooth_fs_2d('iso_2d')
        Lset = np.array([0, 500e-9])
        # code.show_smooth_fs_2d('iso_2d', xxb, ccc, Lset)
        kr = ccc[-1][0]
        n = kr**2 / (2 * np.pi * code.C)
        # ne^2 tau/m
        drude = n * code.Q**2 * Lset[1] / code.HBAR / kr

        sxx0, sxy0, syy0 = code.sigma_2d(1e-6, 101, Lset, xxb, ccc, 1e-7)
        self.assertAlmostEqual(sxy0, 0, delta=100)
        self.assertAlmostEqual(sxx0, drude, delta=100)
        self.assertAlmostEqual(sxx0, syy0, delta=100)
        rxx0 = sxx0 / (sxx0**2 + sxy0**2)

        sxxB, sxyB, syyB = code.sigma_2d(30, 101, Lset, xxb, ccc, 1e-7)
        rxxB = sxxB / (sxxB**2 + sxyB**2)
        ryxB = sxyB / (sxxB**2 + sxyB**2)
        self.assertAlmostEqual(sxxB, syyB, delta=100)
        # Zero MR to 1e-6 accuracy!
        self.assertAlmostEqual(rxxB, rxx0, delta=rxx0 * 1e-6)
        # RH=1/nq to 1e-6 accuracy
        self.assertAlmostEqual(ryxB, 30 / (n * code.Q), delta=abs(ryxB) * 1e-6)

        sxxB2, sxyB2, syyB2 = code.sigma_2d(-30, 101, Lset, xxb, ccc, 1e-7)
        self.assertAlmostEqual(sxxB2, sxxB)
        self.assertAlmostEqual(sxyB2, -sxyB)
        self.assertAlmostEqual(syyB2, syyB)

    @timer
    def test_ICM(self):
        """ Impeded orbital motion; See Hinlopen2022. H-linear MR in the presence
        of hot spots with specific H-linear slope. A rigorous test of the
        lifetime changing with cyclotron motion. """

        xxb, ccc = code.smooth_fs_2d('iso_2d')
        kr = ccc[-1][0]
        n = kr**2 / (2 * np.pi * code.C)
        Lset = np.array([2, 500e-9, 1e-9])
        # code.show_smooth_fs_2d('iso_2d', xxb, ccc, Lset)

        BB = [5, 10]
        rxx = []

        for B in BB:
            sxx, sxy, _ = code.sigma_2d(B, 101, Lset, xxb, ccc, 1e-4)
            rxx.append(sxx / (sxx**2 + sxy**2))

        # See Hinlopen2022 or thesis of Roemer Hinlopen (Eq. 3.19)
        # Const is about 1.18
        # Difference is not from numerical precision,
        # and the numerical value is higher so this is not a matter of an
        #  unsaturated H-linear slope either.
        # Very likely, the remaining 2 % is that there are a few percent of
        #  the FS covered by the hot spots. They are shorted out and
        #  so the effective carrier density n is a bit lower and the slope
        #  a bit higher than the analytical formula for a point hot-spot.
        #  A well known effect explored in the references.
        const = np.pi / 2 / (1 + (np.pi / 2 - 1)**2)
        exp_slope = const / (n * code.Q)
        slope = (rxx[1] - rxx[0]) / 5
        self.assertAlmostEqual(slope, exp_slope, delta=0.02 * exp_slope)

    @timer
    def test_low_field_hall_2d(self):
        """ Big problem. sxy(0)=sxx(0)/1000 even with high quality settings.
        This is enough to really change nH for the first 3 Tesla.
        (rxy only doubles the zero-field value at 0.5 T in reality)
        Tried going for N=2401, err down to 1e-6, but to no avail.

        Write this unittest and start fixing it.
        """

        xxb, ccc = code.smooth_fs_2d('outer_square')
        Lset = np.array([0, 5e-9])
        sss = code.sigma_2d(1e-5, 801, Lset, xxb, ccc, 1e-5)
        self.assertAlmostEqual(sss[1], 0, delta=1000)

        xxb, ccc = code.smooth_fs_2d('inner_square')
        Lset = np.array([0, 5e-9])
        sss = code.sigma_2d(1e-5, 301, Lset, xxb, ccc, 1e-5)
        self.assertAlmostEqual(sss[1], 0, delta=100)

        xxb, ccc = code.smooth_fs_1d('chain')
        Lset = np.array([0, 5e-9])
        sss = code.sigma_1d(1e-5, 301, Lset, xxb, ccc, 1e-5)
        self.assertAlmostEqual(sss[1], 0, delta=10)

    @timer
    def test_sign_hall_effect(self):
        """ Basically I expect 2d pockets to be positive sxy,
        and the 1d pocket zero for isotropic L, and
        the sign changes depending on the anisotropy following
        On1991.

        This issue comes up because I do not trace cyclotron motion
        with direction, I just integrate blindly in 1 direction,
        always assuming hole-like carriers.
        """

        N = 301

        # 1d
        xxb, ccc = code.smooth_fs_1d('chain')
        Lset = np.array([3, 10e-9, 5e-9])
        sss = code.sigma_1d(50, N, Lset, xxb, ccc, 1e-5)
        self.assertLess(sss[1], -10000)

        Lset = np.array([3, 5e-9, 5e-9])
        sss = code.sigma_1d(50, N, Lset, xxb, ccc, 1e-5)
        self.assertAlmostEqual(sss[1], 0, delta=1000)

        Lset = np.array([3, 5e-9, 10e-9])
        sss = code.sigma_1d(50, N, Lset, xxb, ccc, 1e-5)
        self.assertGreater(sss[1], 10000)

        # 2d
        xxb, ccc = code.smooth_fs_2d('inner_square')
        sss = code.sigma_2d(50, N, Lset, xxb, ccc, 1e-5)
        self.assertGreater(sss[1], 10000)

        xxb, ccc = code.smooth_fs_2d('outer_square')
        sss = code.sigma_2d(50, N, Lset, xxb, ccc, 1e-5)
        self.assertGreater(sss[1], 10000)

    @timer
    def test_drude_1d(self):
        """ Drude test for the other half of the code. """

        xxb, ccc = code.smooth_fs_1d('iso_1d')
        Lset = np.array([0, 500e-9])
        # code.show_smooth_fs_1d('iso_1d', xxb, ccc, Lset)

        sxx, sxy, syy = code.sigma_1d(1e-6, 101, Lset, xxb, ccc, 1e-7)

        expect = code.Q**2 * Lset[1] * 2 / code.HBAR / code.A / code.C / np.pi
        self.assertAlmostEqual(syy, expect, delta=100)
        self.assertAlmostEqual(sxx, 0, delta=1e-4)
        self.assertAlmostEqual(sxy, 0, delta=1e-4)

        # This is wctau=8.5 so enormous.
        # Does not change a thing for the result, but it does for the computation.
        sxx, sxy, syy = code.sigma_1d(30, 101, Lset, xxb, ccc, 1e-7)
        self.assertAlmostEqual(syy, expect, delta=100)
        self.assertAlmostEqual(sxx, 0, delta=1e-4)
        self.assertAlmostEqual(sxy, 0, delta=1e-4)

    @timer
    def test_curved_1d(self):
        """ The drude test is a bit lackluster in 1d, so do this too. """

        xxb, ccc = code.smooth_fs_1d('chain')
        Lset = np.array([0, 500e-9])
        # code.show_smooth_fs_1d('iso_1d', xxb, ccc, Lset)

        sxx, sxy, syy = code.sigma_1d(1e-5, 101, Lset, xxb, ccc, 1e-7)

        # The error is zero
        # Because no curved paths or anything that is inaccurate.
        # Sxx is about 5.3e6
        # I did test that sxy goes to zero:
        # It is reduced from 40000 to only -80 at 801 points.
        expect = code.Q**2 * Lset[1] * 2 / code.HBAR / code.A / code.C / np.pi
        self.assertAlmostEqual(syy, expect, delta=expect / 50)
        self.assertAlmostEqual(sxx, 0, delta=expect / 20)
        self.assertAlmostEqual(sxy, 0, delta=sxx / 100)

        # This is wctau=8.5 so enormous.
        # Does not change a thing for the result, but it does for the computation.
        # No test suitable for
        sxx2, sxy2, syy2 = code.sigma_1d(30, 101, Lset, xxb, ccc, 1e-7)
        self.assertAlmostEqual(syy2, syy, delta=expect / 500)
        self.assertLess(sxx2, sxx / 100)
        self.assertGreater(sxx2, sxx / 1000)

        sxx3, sxy3, syy3 = code.sigma_1d(-30, 101, Lset, xxb, ccc, 1e-7)
        self.assertAlmostEqual(sxx3, sxx2)
        self.assertAlmostEqual(sxy3, -sxy2)
        self.assertAlmostEqual(syy3, syy2)

    @timer
    def test_error_estimates(self):
        """ Drude 2d test """

        xxb, ccc = code.smooth_fs_2d('iso_2d')
        Lset = np.array([0, 500e-9])
        # code.show_smooth_fs_2d('iso_2d', xxb, ccc, Lset)
        kr = ccc[-1][0]
        n = kr**2 / (2 * np.pi * code.C)
        # ne^2 tau/m
        drude = n * code.Q**2 * Lset[1] / code.HBAR / kr

        ss, err = code.sigma_with_error(1e-6, 101, Lset,
                                        xxb, ccc, 1e-6, True)
        self.assertAlmostEqual(ss[1], 0, delta=100)
        self.assertAlmostEqual(ss[0], drude, delta=100)
        self.assertAlmostEqual(ss[0], ss[2], delta=100)

        self.assertAlmostEqual(ss[0], drude, delta=err[0] * 2)
        self.assertGreater(abs(ss[0] - drude), err[0] / 2)
        self.assertLess(err[0], 100)

        # Really strict tests:
        #  The error must be small and *representative*.
        #  Whether the error is actually 1e-7 I do not care as much.
        #  (It is about 1e-6 relative instead)
        rxx0 = ss[0] / (ss[0]**2 + ss[1]**2)
        rr, ee = code.convert_rho(ss, err)
        self.assertAlmostEqual(rr[0], rxx0, delta=1e-5)
        self.assertAlmostEqual(rr[0], 1 / drude, delta=ee[0] * 2)

        ssB, errB = code.sigma_with_error(30, 101, Lset, xxb, ccc, 1e-6, True)
        rrB, eeB = code.convert_rho(ssB, errB)
        self.assertAlmostEqual(ssB[0], ssB[2], delta=100)

        # Zero MR to large accuracy!
        # Again, reasonably *representative* error
        self.assertAlmostEqual(rrB[0], rr[0], delta=rr[0] * 3e-6)
        self.assertAlmostEqual(rrB[0], 1 / drude, delta=eeB[0] * 2)
        self.assertGreater(abs(rrB[0] - 1 / drude), eeB[0] / 6)
        self.assertLess(eeB[0], 100)

        # RH also
        # Again, reasonably *representative* error
        self.assertAlmostEqual(rrB[1], 30 / (n * code.Q), delta=eeB[1] * 2)
        self.assertGreater(abs(rrB[1] - 30 / (n * code.Q)), eeB[1] / 10)


if __name__ == '__main__':
    unittest.main(exit=False)
    print(flush=True)
    plt.show()
