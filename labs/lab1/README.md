# Workshop I

## How is it distributed?

* The main code is a jupyter notebook with name `lab1.ipynb` that needs a configuration file with name `lab1_config.json`.

* First of all, it is necessary to download all the photos of the cohort and upload them in the local directory `imgs/cohort/`. The main jupyter will load all the images to generate the result.
* There is a folder with name `unsupervised/` that contains a subfolder with name `dim_red/` with the implementations of SVD and PCA.
* There is a `Pipfile` and a `Pipfile.lock` files to manage modules and dependencies.
* There is a file with name `app.py` for the http server (but it is currently in process).