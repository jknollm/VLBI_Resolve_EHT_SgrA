## Resolving Horizon-scale Dynamics of Sagittarius A*

This repository contains the code to perform a dynamic reconstruction of Sagittarius A* (SgrA*) from Event Horizon Telescope (EHT) data.
It was used to obtain the results in the paper "Resolving Horizon-Scale Dynamics of Sagittarius A*".

This version of the code only relies on the EHT data, not the additional ALMA light curves, as they have not been made public.

The full reconstruction takes several days even on a large machine, but intermediate results are provided and can also be obtained on smaller machines in short amounts of time.

### Data
The data in this repository is a copy of the data published by the EHT Collaboration and can be found here:

https://doi.org/10.25739/m140-ct59


### Requirements 

\```python
pip install git+https://gitlab.mpcdf.mpg.de/ift/resolve.git

pip install ehtim

pip install mpi4py

pip install git+https://github.com/liamedeiros/ehtplot.git
\```

### Run Code
In order to run the reconstruction use the following command. Please adapt N according to the number of available processes. Note that N>32 does not improve performance and we recommend N to be a power of 2 in order to ensure balanced loads.

\```python
mpirun -np N python reconstruction.py cfg/EHT_SgrA_dynamic_April_6th_2017.cfg
\```



