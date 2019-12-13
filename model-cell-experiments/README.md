# Application to electrical model cells

All electrical model cell experiments are within this directory.


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

