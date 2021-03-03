# Voltage-Clamp Experiment Model

A detailed mathematical model of a voltage-clamp experiment with imperfect patch-clamp amplifier compensations

## Release Notes

### March 2021
The latest version of this repository includes supercharging compensation, following the equations in Chapter 6 of [Chon Lok Lei's thesis](https://ora.ox.ac.uk/objects/uuid:528c2771-ae4f-4f3c-b649-44904acdf259)._

### April 2020
This repo contains all data and code for reproducing the results in the paper "*Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments*" by Chon Lok Lei, Michael Clerx, Dominic Whittaker, David Gavaghan, Teun de Boer, and Gary Mirams.
[doi:10.1098/rsta.2019.0348](https://doi.org/10.1098/rsta.2019.0348).
A permanently archived version (without supercharging) for reproducing the results in the paper is available [as a tagged release](https://github.com/CardiacModelling/VoltageClampModel/releases/tag/v1) or on Zenodo at <https://doi.org/10.5281/zenodo.3754184>.

### Installing

To run the code to generate the data and figures shown in the paper, please follow these steps first:

1. Clone [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation) to the relative path `../hERGRapidCharacterisation`.
2. To setup, either run (for Linux/macOS users)
```console
$ bash setup.sh
```
or
install [PINTS](https://github.com/pints-team/pints) and [Myokit](http://myokit.org) manually with Python 3.5+.


### Results

#### Section 3: Validating the mathematical model with electrical model cell experiments

- [model-cell-experiments](./model-cell-experiments): Application of the voltage-clamp experiment model to electrical model cell experiments.
  - contains results and code to reproduce Figures 4, 5; Table 2; Figures S4, S5; Tables S1, S2.

#### Section 4: Application to variability in CHO-hERG1a patch-clamp data

- [herg-syn-study](./herg-syn-study): Synthetic studies with the voltage-clamp experiment model.
  - contains results and code to reproduce Figure S6.

- [herg-real-data](./herg-real-data): Application of the voltage-clamp experiment model to data in [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation).

  - Contains results and code to reproduce Hypothesis 2: Figures 7, 8; Figures S7, S8, S9; Tables S3, S4.
  - Also contains results and code that are mentioned but not shown in the paper: full voltage-clamp experiment model fitting to hERG data; independent kinetics with independent artefacts 'Hypothesis 3'.

### Supporting files

- [lib](./lib): Contains all modules/utility functions.

- [mmt-model-files](./mmt-model-files): [Myokit](http://myokit.org/) model files, contains IKr model and voltage clamp experiment model etc.

- [protocol-time-series](./protocol-time-series): Contains protocols as time series, stored as `.csv` files, with time points (in `s`) and voltage (in `mV`)

- `data`, `data-autoLC`, `manualselection`, `qc`: symbolic links to [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation)

- [cellml](./cellml): [CellML](https://www.cellml.org/) version of the models in [mmt-model-files](./mmt-model-files).


## Acknowledging this work

If you publish any work based on the contents of this repository please cite ([CITATION file](CITATION)):

Lei, C.L., Clerx, M., Whittaker, D.G., Gavaghan D.J., de Boer, T.P. and Mirams, G.R.
(2020).
[Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments](https://doi.org/10.1098/rsta.2019.0348).
Philosophical Transactions of the Royal Society A, 378: 20190348.
