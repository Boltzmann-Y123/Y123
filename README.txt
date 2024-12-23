This readme provides an overview of the content in this folder.
Written by the author, Roemer Hinlopen.
PLEASE recall that reproducing or using this code requires attribution to the paper, 
	see the LICENSE file.

In short:
	The figures in the paper are numbers 30, 40, 50 in the "04 figs" folder.
	The code to generate these are the 07.py, 08.py, 11.py python files respectively.
	The data is in the 02 & 03 folders with settings listed at the top of each file.
	A coarse overview is described below structuring the project as a whole. 
		For detail about each step please refer to the documentation of functions
		and to the code itself. Significant effort is put into keeping it legible.

////////////////////////////////
// Steps to obtain the Fermi surface
////////////////////////////////

Folder "01 Carrington2007 FS data" contains the digitized Fermi 
	surface from 10.1103/PhysRevB.76.140508 at one kz value.
	These files are barely human legible.
	They are decoded and plotted by the code below.

The input for and smoothing of the Fermi surface. 
	Figures below generated from 02.py.
01: This is the Fermi surface data as digitized from Carrington2007.
	The paper produces a view of the FS at one Kz value from DFT.
02: Same but instead of (kx, ky) show (phi, r).
03: Artificial x & y noisy data is smoothed with a local polynomial fit.
04: The derivative is still wiggly. Smooth it in a second round.
05: Integrate the smoothed derivative to obtain the final result
	and compare to the true noiseless function.
	This is doubly smoothed.

Next apply smoothing to the FS.
	This defines the necessary input to the conductivity calculations.
	Figures below generated from 03.py
10: Input (blue dots) and doubly smoothed (red line) inner square Fermi pocket.
11: The L orbit if one used 5m0 and 0.1 ps isotropic mass and tau.
	This shows that the mean free path is smoothly varying and
	only has small remnant wiggles after double smoothing.
	Recall from Ong1991 that sigma_xy at low magnetic fields is 
	determined by the square area of such L orbits.
12: Similar to 10 for the outer square, which has mixed planar (east and west)
	and chain (north and south) character in a pocket which is 2d.
13: Same as 11 for the outer square.
14: Similar to 10 for the 1d pocket. The middle segment originates from the plane,
	the left and right parts from the chain.


////////////////////////////////
// Planar carrier density
////////////////////////////////


For the planar carrier density
	Figure and text generated from 04 planar.py
20: The complete Fermi surface. In black the outline of the 
	pre-hybridisation planar Fermi pockets. The planar carrier 
	density of the model is extracted from the area
	enclosed by the two black curves via the Fermi volume
	taking into account spin degeneracy. 
	This assumes no pseudogap and thus 1+p end of the p to 1+p crossover.
21: Numerical values for the planar carrier density.
	Also computed is the carrier density from the inner and outer square
	pockets post-hybridisation. Doing so is wrong and results in electron doping
	because a sizble number of holes hybridised with the chain band are neglected.


////////////////////////////////
// Conductivity code
////////////////////////////////


Code 04_compute.py contains all the code from FS smoothing to conductivity calculation in 1 file.
	All details of the computation are herein.
	There is innovation here in the numerical algorithms,
	namely in the way that the inner two integrals of Eq2 of the SI of the paper
	are only computed between subsequent points and then convoluted. This saves
	10-100x on execution time. Compilation with numba saves 10-100x on execution time.
	Finally, on an individual segment time-ordered integration is used which is crucial
	to adapt the stepsize and zoom in when wc*tau is negligible as B->0. 
	The latter allows this code to go down to about 1 micro Tesla before numerical precision
	and execution time really blow up to unfeasible levels -- which is less than earth's magnetic field.
	These techniques are described also in an appendix of the thesis of Roemer Hinlopen,
	though in the more general context of ADMR (meaning Kz corrugation is not neglected and the magnetic field
	is in arbitrary direction). 

	These are powerful techniques to combined speed up Boltzmann transport code by >1000x.


Code 04_tests contains all sorts of verification for the code.
	Tests symmetries which must be present
	Tests sign and size of the Hall effect
	Tests conductivity against Drude theory
	Tests conductivity against impeded cyclotron motion 
		(see the paper Hinlopen2022, Phys Rev Res at 10.1103/PhysRevResearch.4.033195)
	Tests the errors produced are sensible
	All tests passed at the time of uploading using these versions:
		Python 3.12.7
		Numba 0.60.0
		NumPy 2.0.2
		Matplotlib 3.9.2
	These are the only dependencies in the project.
	A reminder that no guarantees are given about the code using the same or other versions
		of these dependencies as stated in the license file.


////////////////////////////////
// Data files
////////////////////////////////

Then the results of the program, starting with magnetic field sweeps.
	Code 05.py is a script with near the bottom 4 sections which are the settings
	used to generate datafiles 0001, 0002, 0003 and 0004 in "02 output Bsweep".


Code 10.py scans anisotropy instead of magnetic field.
	The planar mean free path is kept constant and anisotropy
	ratio in mean free path is scanned.

Summarised in terms of data:
	"02 output Bsweeps" contains human legible magnetic field sweeps.
	Each file contains one row per magnetic field value, listing the 
	conductivity contributions of each band for each component (9 in total), 
	each with their estimated numerical error. 19 columns total.
	The exact settings used are listed at the top of the file.
		0001 isotropic L
		0002 dirty chains Lchain/Lplane=1/3
		0003 clean chains Lchain/Lplane=2
		0004 dirty chains Lchain/Lplane=1/2
		All between 0 and 5000 Tesla
	"03 output ani sweep" contains human legible chain purity sweeps at constant B.
		0001 at 60 T from Lc/Lp=0.02 to 29.5


////////////////////////////////
// Result graphs
////////////////////////////////

The results of the "show" codes to plot these data are all put in 05_recent.
	Each time one of these runs, it removes all content from that folder to 
	avoid clutter. Copies of the results are put in "04 figs" as detailed below.

Code 06.py shows the full result of any one of these cases, whichever is entered
	in the main function call at the bottom of the file.
	The default is file 0001 (isotropic L)

Code 07.py is a slightly modified version of 06.py to show specifically
	the behaviour up to 100 T in data 0001, namely isotropic L.
	This generates a figure in the supplement.
In 04/figs:
	30 shows the main result reproduced in the supplement of the paper for isotropic L
	31 a view of the Fermi surface with isotropic L.
		In the paper, this figure is pasted as inset into 30.
	32-34 show the mean free path orbits. Notice that with isotropic L
		the area enclosed by the 1d pocket must necessarily vanish,
		meaning sigma_xy_1d=0 as well.

Code 08.py is another modification of 06.py to show specifically
	the high field behaviour of data 0001 and 0003, namely
	isotropic L and somewhat dirty chains (common at low T where 
	disorder blocks the chains disproportionately).
	This generates a figure in the supplement.
In 04/figs:
	40 shows the main result reproduced in the supplement of the high B dependence
	41 shows additional details about these results, namely in the top row
		for isotropic L and in the bottom row for dirty chains.
		This includes the 9 conductivity components, 
		the resistivities with field and the anisotropy ratio with field.
	42 & 43 shows the isotropic & anisotropic L on the Fermi surface
	44 through 49 shows the L orbits of the 3 pockets in both isotropic and anisotropic cases.
	
Code 10.py scans anisotropy instead of magnetic field.
	The planar mean free path is kept constant and anisotropy
	ratio in mean free path is scanned 
	The main result swept at 60 T is in "03 output ani sweep" #0001.
	Code 11.py is a modification of 06.py to show the anisotropy sweep.
	Data up to Lc/Lp=30 is provided in the data file.
In 04/figs:
	50 main result reproduced in the supplement of the paper.
	51 The Fermi surface and mean free path across it at Lc/Lp=0.3
		showing how the anisotropy is applied to the original
		planar and chain Fermi surfaces. As a result, they are mixed
		post-hybridisation and create sizable L anisotropy on each pocket.
	52-54 the corresponding L orbits. See Ong1991, the area of such a curve
		corresponds to the size of sigma_xy in he low-field limit.

