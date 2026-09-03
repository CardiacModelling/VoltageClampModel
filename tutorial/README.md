# Voltage clamp model tutorials

In these notebooks we construct (1) a model of a patch-clamp experiment with various experimental artefacts, and (2) a model of the corrections applied by patch-clamp amplifiers to mitigate these effects.
The initial exposition draws on a book chapter by [Sigworth (1995a)](https://doi.org/10.1007/978-1-4419-1229-9_4), but replaces Laplace-domain analysis with an ODE formulation.

To view the notebooks, use the GitHub or nbviewer links below, or clone the repository and run Jupyter notebook locally.
Running locally will require the dependencies from `requirements.txt`.

[↩ Back to the main repository](https://github.com/CardiacModelling/VoltageClampModel)


## 1. Modelling patch-clamp experiments 
[![github](./img/github.svg)](./1-modelling-patch-clamp.ipynb)
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/1-modelling-patch-clamp.ipynb)

The first notebook describes the uncompensated patch-clamp set up, and derives an ODE model from the electrical schematics.

## 2. Modelling electronic compensation
[![github](./img/github.svg)](./2-compensation.ipynb)
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/2-compensation.ipynb)

The model is updated to include the compensation circuitry commonly used in patch-clamp amplifiers.

## 3. Simulating a manual patch clamp experiment 
[![github](./img/github.svg)](./3-simulations.ipynb) 
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/3-simulations.ipynb)

We walk through and simulate the early steps of a manual patch-clamp experiment.

## 4. Simplified models 
[![github](./img/github.svg)](./4-simplifications.ipynb) 
[![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/4-simplifications.ipynb)

In the final notebook, we derive simplified models and compare with previous work.

## Resources

- [![github](./img/github.svg)](./symbols.ipynb) 
  [![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/symbols.ipynb)
  **Names and symbols** A table of all symbols, their meanings, and names used in other publications.

- [![github](./img/github.svg)](./tour.ipynb) 
  [![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/tour.ipynb)
  **Default parameter values** and comments on how to reparameterise the model, are given in the Model Tour.

- [![github](./img/github.svg)](./references.ipynb) 
  [![nbviewer](./img/nbviewer.svg)](https://nbviewer.jupyter.org/github/CardiacModelling/VoltageClampModel/tree/main/tutorial/references.ipynb)
  **References** and further reading.

Finally, there are [several appendices](./appendix), which provide background on electronics and details of the model derivation.

