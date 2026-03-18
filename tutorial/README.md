# Voltage clamp model tutorials

In these notebooks we retrace the steps taken in the supplement to [Lei et al., 2020](https://doi.org/10.1098/rsta.2019.0348) and [2025](https://doi.org/10.1002/advs.202500691), and construct (1) a model of a patch-clamp experiment with various experimental artefacts, and (2) a model of the corrections applied by patch-clamp amplifiers to mitigate these effects.
The initial exposition draws on a book chapter by [Sigworth (1995a)](https://doi.org/10.1007/978-1-4419-1229-9_4), but replaces Laplace-domain analysis with an ODE formulation.

To view the notebooks, use the GitHub or nbviewer links below, or clone the repository and run jupyter notebook locally.
Running locally will require the dependencies from `requirements.txt`.

A list of references and further reading is provided [here](./references.ipynb).

[↩ Back to the main repository](https://github.com/CardiacModelling/VoltageClampModel-new)


## Modelling patch-clamp experiments 
[![github](./img/github.svg)](./1-modelling-patch-clamp.ipynb)
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel-new/tree/main/tutorial/1-modelling-patch-clamp.ipynb)

The first notebook describes the uncompensated patch-clamp set up, and derives an ODE model from the electrical schematics.

## Modelling electronic compensation
[![github](./img/github.svg)](./2-compensation.ipynb)
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel-new/tree/main/tutorial/2-compensation.ipynb)

The model is updated to include the compensation circuitry commonly used in patch-clamp amplifiers.

## Simulating a manual patch clamp experiment 
[![github](./img/github.svg)](./3-simulations.ipynb) 
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel-new/tree/main/tutorial/3-simulations.ipynb)

We walk through and simulate the early steps of a manual patch-clamp experiment.

## Simplified models 
[![github](./img/github.svg)](./4-simplified.ipynb) 
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel-new/tree/main/tutorial/4-simplified.ipynb)

In the final notebook, we derive simplified models and compare with previous work.

---

## Appendices

Finally, there are several appendices:

- [Appendix A](./appendix-a/README.md) adds details on electronics and filters.
- [Appendix B](./appendix-b/README.md) looks at details of the model derivation.
- [Appendix C](./appendix-c/README.md) **provides default parameter values**, and amplifier-specific ones.
- [Appendix D](./appendix-d/README.md) discusses remaining sources of error.
- [Appendix E](./appendix-e/README.md) looks at Rs and Cm estimates.

