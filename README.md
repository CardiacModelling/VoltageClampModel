# Voltage-Clamp Experiment Model

A detailed mathematical modelling of voltage-clamp experiment.
This repo contains all data and code for reproducing the results in the paper "*Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments*" by Chon Lok Lei, Michael Clerx, Dominic Whittaker, David Gavaghan, Teun de Boer, and Gary Mirams.


### Prerequisite

1. Clone [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation) to the relative path `../hERGRapidCharacterisation`.
2. To setup, either run (for Linux/macOS users)
```console
$ bash setup.sh
```
or
install [PINTS](https://github.com/pints-team/pints) and [Myokit](http://myokit.org) manually with Python 3.5+.


### Results

#### Section 3: Validating the mathematical model with electrical model cell experiments
[model-cell-experiments](./model-cell-experiments): Application of the voltage-clamp experiment model to electrical model cell experiments.

It contains results and code to reproduce Figures 4, 5; Table 2; Figures S4, S5; Tables S1, S2.

#### Section 4: Application to variability in CHO-hERG1a patch-clamp data
[syn-same-kinetics](./syn-same-kinetics): Synthetic studies with the voltage-clamp experiment model.

It contains results and code to reproduce Figure S6.

[fit-same-kinetics](./fit-same-kinetics): Application of the voltage-clamp experiment model to data in [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation).

It contains results and code to reproduce Figures 7, 8; Figures S7, S8, S9; Tables S3, S4.

### Supporting files

[lib](./lib): Contains all external Python libraries (require manual installation, see [README](./lib/README.md)) and other modules/utility functions.

[mmt-model-files](./mmt-model-files): [Myokit](http://myokit.org/) model files, contains IKr model and voltage clamp experiment model etc.

[protocol-time-series](./protocol-time-series): Contains protocols as time series, stored as `.csv` files, with time points (in `s`) and voltage (in `mV`)

`data`, `data-autoLC`, `manualselection`, `qc`: symbolic links to [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation)


## Acknowledging this work

If you publish any work based on the contents of this repository please cite:

Lei, C.L., Clerx, M., Whittaker, D.G., Gavaghan, D.J., de Boer, T.P. and Mirams, G.R.
(2019).
[Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments](https://doi.org/10.1101/2019.12.20.884353).
_bioRxiv_, 2019.12.20.884353.
