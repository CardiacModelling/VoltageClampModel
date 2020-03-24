# Application to real hERG data

[Figure 7](figs/rmsd-hist-fix-kinetics-simvclinleak-scheme3-part1.pdf)

1. Run `fit-fix-kinetics-simvclinleak.sh` to get initial guesses for the artefact parameters.
2. Run `fit-simvclinleak-scheme3-parallel.py` to run fits with Eq. (4.14) using Algorithm S1.
3. Run `compute-rmsd-hist-fix-kinetics-simvclinleak-scheme3.py` to compute all the RRMSE values.
4. Run `plot-rmsd-hist-fix-kinetics-simvclinleak-scheme3-part1.py` to generate Figure 7.

[Figure 8](figs/herg25oc1-simvclinleak-scheme3-fitted-parameters.pdf)

1. Generate Figure 7
2. Run `plot-fitted-parameters-fix-kinetics-simvclinleak-scheme3.py` to generate Figure 8.


