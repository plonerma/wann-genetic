# Changes to the codebase

The following changes have already been made.


### Replaced MPI-driven parallel computation by multiprocessing module

Using the multiprocessing module allows much more concise code.
Employing the MPI-protocol wouldn't be beneficial in the computation
environments that I will be using.


### Unified cli scripts for training wann and neat networks

Files `train.py` and `trainer.py` will replace `wann_train.py` and
`neat_train.py`. The latter two share a lot of code and will be deprecated.
