# Voltage-Clamp Experiment Model

A detailed mathematical modelling of voltage-clamp experiment.


### Prerequisite

1. Clone [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation) to the relative path `../hERGRapidCharacterisation`.
2. To setup, run
```console
$ bash setup.sh
```


### Results

[syn-same-kinetics](./syn-same-kinetics): Synthetic studies with the voltage-clamp experiment model.

[fit-same-kinetics](./fit-same-kinetics): Application of the voltage-clamp experiment model to data in [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation).


### Supporting files

[lib](./lib): Contains all external Python libraries (require manual installation, see [README](./lib/README.md)) and other modules/utility functions.

[mmt-model-files](./mmt-model-files): [Myokit](http://myokit.org/) model files, contains IKr model and voltage clamp experiment model etc.

[protocol-time-series](./protocol-time-series): Contains protocols as time series, stored as `.csv` files, with time points (in `s`) and voltage (in `mV`)

`data`, `data-autoLC`, `manualselection`, `qc`: symbolic links to [hERGRapidCharacterisation](https://github.com/CardiacModelling/hERGRapidCharacterisation)

