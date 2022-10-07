# Solar-inelastic-WIMPs
IceCube analysis production code

This repository contains the software framework for Raffi's solar inelastic WIMPs analysis. In a nutshell:

- We are looking at neutrino signals from WIMP annihilation inside the Sun.
- The scalar nature of the WIMPs is described by the scotogenic minimal dark matter model with a number of model parameters.
- Certain combinations of model parameters (i.e. scenarios) allow for inelastic scattering of WIMPs with solar nuclei, which boosts the signal neutrino flux and makes the scotogenic model accessible to indirect detection.
- We investigate ~1700 scenarios and use this framework to exclude some of them.

Analysis motivation, theoretical background, results, methods, locations of data files, links to presentations, etc., can be found on the [Analysis wiki](https://wiki.icecube.wisc.edu/index.php/Solar_inelastic_WIMPs_analysis).


## Installation

Clone this repo and just execute the setup script:

`setup.py install --user`

Note that this code runs with data files that are not provided here (the locations of these files are given in the analysis wiki). Access to IceCube software and computing infrastructure is required to use this software and to reproduce the analysis results.


## Reproducing results

This repository contains all necessary configuration files to run the analysis chain, provided that you have access to the data files listed in the analysis wiki. If the data file structure is changed, the selection configuration files

`config/int_config.py`
`config/osc_config.py`

have to be modified accordingly. Otherwise, the only file to be manipulated is

`tests/locations.json`

which contains the locations of the data directory (should be `/data/ana/BSM/solar_inelastic_WIMPs/`) and the desired output directory, as well as the reproduction test script

`repro_test.py`.

You can run the analysis chain for a specific scenario (default 0849.4492 of batch 001 with the nominal data set) by simply executing the script. Note that the script may run for several hours! The following files will appear in the specified output directory:

```
INT_BPDF_histogram(0.0020-0.0200).npy
INT_BPDF_KDE_evalfine(0.0020-0.0200)-intp.pkl
INT_BPDF_KDE_evalfine(0.0020-0.0200).npy
INT_SPDF_histogram.npy
INT_SPDF_KDE_evalfine-intp.pkl
INT_SPDF_KDE_evalfine.npy
OSC_BPDF_histogram(0.0420-0.0200).npy
OSC_BPDF_KDE_evalfine(0.0420-0.0200)-intp.pkl
OSC_BPDF_KDE_evalfine(0.0420-0.0200).npy
OSC_SPDF_histogram.npy
OSC_SPDF_KDE_evalfine-intp.pkl
OSC_SPDF_KDE_evalfine.npy
TS_background_00000.npy
sensitivity.npy
```

Alternatively, you can pass a random state seed with

`$ ./repro_test.py --seed [some integer]`

to reproduce a specific result. Use `--seed 1000` to compare with the
md5 checksums in

`tests/001-0849.4492-nominal_seed1000_md5sums.txt`.

If you want to test a different scenario (and/ or a data set that is not the nominal one), you can specify it with the arguments `--batch`, `--name`, and `--set`. You find all available scenarios at

`/data/ana/BSM/solar_inelastic_WIMPs/scenarios/`.
