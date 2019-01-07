# documentation and stuff

Right now everything is inside a google docs: https://docs.google.com/document/d/1_r8AYzP6EbS9Ynvjod2h4EX_h4W27GZN_Cb1dbOyX30/edit#


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


# notebook

In order to run notebooks load your virtual environment, install the pip dependencies, cd to notebooks and run
