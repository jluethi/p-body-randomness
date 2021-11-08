# P-body Localization: Monte Carlo Simulations
This repository contains the Monte Carlo simulations for P-body localization testing the observed localization against a random localization of P-bodies within the cytoplasm of HeLa cells. Using nearest neigbhor measurements, we compare this and show that the distribution of P-bodies within the cytoplasm is very similar to a random distribution.

# Overview & coordination

All the work has been done during the 2019 block course of the Systems Biology Graduate school at UZH & ETH Zurich.
The coordination is inside a google docs: https://docs.google.com/document/d/1_r8AYzP6EbS9Ynvjod2h4EX_h4W27GZN_Cb1dbOyX30/edit#

# data

- data/input_data is part of git and contains the small data set where we test on
- data/processed is NOT part of the repository (see .gitignore). All intermediate data can be stored there

# setup

    mkvirtualenv p-body-randomness
    pip install -r requirements.txt
    ipython kernel install --user --name=p-body-randomness
    python setup.py develop
    # now you can start develop (source code is in src/p_body_randomness)
    # or you can cd to notebooks/ and use your project code:
    cd notebooks
    jupyter notebook

# Collaborators
- [Joel Lüthi](https://github.com/jluethi)
- [Moritz Schaefer](https://github.com/moritzschaefer)
- [Marija Dmitrijeva](https://github.com/marydmit)
