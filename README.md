# Voltage clamp model

This repository contains up-to-date reference versions of the voltage clamp model used in Lei et al. 2020, 2025, and other publications.
These models can be used to simulate manual or planar patch-clamp experiments in voltage-clamp mode.

_Note: This URL used to host the data for the 2020 Lei et al. publication, which has been moved to https://github.com/CardiacModelling/VoltageClampModel2020_

## The latest models

We recently updated the voltage clamp model with improved filtering of input (stimulus filter) and output (Bessel filters), and an improved time-delay in the series resistance compensation pathway.

In addition to this full-featured model, we now also provide a series of increasingly simplified models:

- The "Level 0" model includes all features.
- The "Level 1" model simplifies this, by replacing all filters with first-order approximations.
- "Level 2" further assumes an ideal measuring op-amp without stray capacitance.
- "Level 3" removes fast capacitance currents and their correction.
- "Level 4" removes all filtering entirely.
- "Level 5" assumes perfect slow capacitance cancellation (but imperfect series resistance compensation).

All models are provided in Myokit ([models-mmt](./models-mmt)) and CellML ([models-mmt](./models-mmt)) formats.

### Which model should I use?

Levels 0, 1, and 2 can recreate the _fast_ artefacts seen in patch clamp experiments, and are very useful to understand the patch clamp process.

To fit experimental data, these fast artefacts are less important, and so **level 3 is good to match data from the fastest currents**.
For slower currents, levels 4 or 5 can be used.

## Tutorials

To understand these models, we provide [four tutorial notebooks](./tutorial/README.md).

The first derives a basic model of a patch clamp amplifier, and the second adds compensation and filtering, leading to the "Level 0" model.
In the third notebook, this model is used to simulate the early stages of a (manual) patch-clamp experiment.

The final notebook derives the simplifications, and shows how they relate to our previous work (Lei et al., 2020 and 2025).

## Previous versions

### Lei et al. 2000

The first published version was in

> Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments.
  Lei, C.L., Clerx, M., Whittaker, D.G., Gavaghan D.J., de Boer, T.P. and Mirams, G.R. (2020).
  Philosophical Transactions of the Royal Society A, 378: 20190348.
  https://doi.org/10.1098/rsta.2019.0348

BibTeX entry:
```
@article{Lei2020Variability,
  author = {Lei, Chon Lok  and Clerx, Michael  and Whittaker, Dominic G.  and Gavaghan, David J.  and de Boer, Teun P.  and Mirams, Gary R. },
  title = {Accounting for variability in ion current recordings using a mathematical model of artefacts in voltage-clamp experiments},
  journal = {Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences},
  volume = {378},
  number = {2173},
  pages = {20190348},
  year = {2020},
  doi = {10.1098/rsta.2019.0348}
}
```

Code: https://github.com/CardiacModelling/nav-artefact-model

### Lei et al. 2025

The "supercharging" or "prediction" pathway was added in

> Resolving Artifacts in Voltage-Clamp Experiments with Computational Modeling: An Application to Fast Sodium Current Recordings.
  Lei, C.L., Clark, A.P., Clerx, M., Wei, S., Bloothooft, M., de Boer, T.P., Christini, D.J., Krogh-Madsen, T., and Mirams, G.R. (2025).
  Advanced Science, 12, 30: e00691.
  https://doi.org/10.1002/advs.202500691

BibTeX entry:
```
@article{Lei2025Artifacts,
  author = {Lei, Chon Lok and Clark, Alexander P. and Clerx, Michael and Wei, Siyu and Bloothooft, Meye and de Boer, Teun P. and Christini, David J. and Krogh-Madsen, Trine and Mirams, Gary R.},
  title = {Resolving Artifacts in Voltage-Clamp Experiments with Computational Modeling: An Application to Fast Sodium Current Recordings},
  journal = {Advanced Science},
  volume = {12},
  number = {30},
  pages = {e00691},
  year = {2025},
  doi = {https://doi.org/10.1002/advs.202500691}
}
```

Code: https://github.com/CardiacModelling/VoltageClampModel2020

## Acknowledging this work

If you publish any work based on the contents of this repository please cite the papers listed above, or use the information in [our CITATION.cff](CITATION.cff).

