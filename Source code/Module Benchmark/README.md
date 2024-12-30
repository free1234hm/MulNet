# Dependencies

Several scientific python packages are required to run the code. The easiest way to do this is to use the python 3.X Anaconda distribution (https://www.continuum.io/downloads).
* numpy
* json
* Cython
* git+https://github.com/jfrelinger/cython-munkres-wrapper

The Cython code for the evaluation metrics should also be compiled. For this, change directory to /lib and run `python setup.py build_ext --inplace`

The Cython munkres library should be installed using e.g. pip install git+https://github.com/jfrelinger/cython-munkres-wrapper

After installation, run module_benchmark.py in the Four_scores/scripts folder.

The code is based on the Jupyter Notebook implementation by Saelens et al. (PMID: 29545622), which can be accessed at www.github.com/saeyslab/moduledetection-evaluation.
