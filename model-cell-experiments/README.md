# Application to electrical model cells

All electrical model cell experiments are within this directory.

[Figure 4](figs/simulate-all-staircase.pdf): Run `simulate-all.py` with argument `staircase`.

[Figure 5A](figs/predict-mc3nocomp-staircase.pdf), [B](figs/predict-mc3nocomp-ap-lei.pdf): Run `predict-mc3-all.sh`.

Table 2, S1: Run `fit-mc.py` and `fit-mc3.py` to obtain the fitting results. Run `print-tex-table.py` to generate the table.

[Figure S4A](figs/simulate-all-mcnocomp-staircase.pdf),
[B](figs/simulate-all-mc3nocomp-staircase.pdf),
[C](figs/simulate-all-mcauto-staircase.pdf),
[d](figs/simulate-all-mc3auto-staircase.pdf): Run `simulate-all.sh`.

[Figure S5A](figs/predict-mcnocomp-staircase.pdf), [B](figs/predict-mcnocomp-ap-lei.pdf): Run `predict-mc.py` with argument `staircase` (A) and `ap-lei` (B).

Table S2: To generate results for Table S2, run `fit-mc3simvc.py`.


### Forward simulations

[simulate-mcnocomp.py](./simulate-mcnocomp.py): Forward simulate Type I Model Cell no compensations experiment.
[simulate-mc3nocomp.py](./simulate-mc3nocomp.py): Forward simulate Type II Model Cell no compensations experiment.
[simulate-mcauto.py](./simulate-mcauto.py): Forward simulate Type I Model Cell HEKA automatic estimated compensations experiment.
[simulate-mc3auto.py](./simulate-mc3auto.py): Forward simulate Type II Model Cell HEKA automatic estimated compensations experiment.

All of them run with argument `[str:which_sim]`: `staircase`, `sinewave`, `ap-lei`.


### Parameter inference

[fit-mc.py](./fit-mc.py): Parameter inference for Type I Model Cell no compensations experiment.
[fit-mc3.py](./fit-mc3.py): Parameter inference for Type II Model Cell no compensations experiment.
[fit-mc3simvc.py](./fit-mc3simvc.py): Parameter inference for Type II Model Cell no compensations experiment with simplified voltage-clamp experiment model.


### Predictions using inferred parameters

[predict-mc.py](./predict-mc.py): Predictions from fitted model using [fit-mc.py](./fit-mc.py).
[predict-mc3.py](./predict-mc3.py): Predictions from fitted model using [fit-mc3.py](./fit-mc3.py).
[predict-mc3simvc.py](./predict-mc3simvc.py): Predictions from fitted model using [fit-mc3simvc.py](./fit-mc3simvc.py).

All of them run with argument `[str:which_predict]`: `staircase`, `sinewave`, `ap-lei`.


### Directories

[data](./data): Contains raw bundle dat data from HEKA PatchMaster.
[out](./out): Contains fitting results and parameters.
[figs](./figs): Contains all output figures.

