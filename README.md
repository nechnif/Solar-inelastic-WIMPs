# Solar-inelastic-WIMPs
IceCube analysis production code

This repository contains the software framework for Raffi's solar inelastic WIMPs analysis. In a nutshell:

-We are looking at neutrino signals from WIMP annihilation inside the Sun.
-The scalar nature of the WIMPs is described by the scotogenic minimal dark matter model with a number of model parameters.
-Certain combinations of model parameters (i.e. scenarios) allow for inelastic scattering of WIMPs with solar nuclei, which boosts the signal neutrino flux and makes the scotogenic model accessible to indirect detection.
-We investigate ~1700 scenarios and use this framework to exclude some of them.

Analysis motivation, theoretical background, results, methods, locations of data files, links to presentations, etc., can be found on the [Analysis wiki](https://wiki.icecube.wisc.edu/index.php/Solar_inelastic_WIMPs_analysis).


## Installation

Clone this repo and just execute the setup script:

`setup.py install --user`

Note that this code has dependencies in terms of data files and locations given in the analysis wiki. Access to IceCube software and computing infrastructure is required to use this software and to reproduce the analysis results.
