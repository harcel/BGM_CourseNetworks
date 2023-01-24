# Bayesian Generative Modeling of Course Networks
This repository contains code and an example notebook of how to model student results in networks of courses (courses taken by partly overlapping student populations), as described in Haas, Caprani and Van Beurden (submitted).

## Packages and environment
The code uses PyMC v4, ArViZ, numpy and pandas (and matplotlib and seaborn for visualizations).


## Open Data and your own data
The example notebook (ModelingCourseNetworks.ipynb) starts with creating mock data and modeling that. It then moves on to real-world data. It is based on data that is [publicly available from the Open Univeristy](https://analyse.kmi.open.ac.uk/open_dataset), but has also been extensively tested on data from the University of Amsterdam, with very similar results. The code base assumes that data for the example notebook is in a data folder in the main directory of the repository. Make sure to put there like suggested by the example notebook.


[Marcel Haas](mailto:datascience@marcelhaas.com), Jan 2023
