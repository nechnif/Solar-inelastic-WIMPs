#!/usr/bin/env python3
import os
import shutil
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from solarinelastic import Model
from solarinelastic import Selection
from solarinelastic.frequentist import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed',  nargs='?', default=None)
    parser.add_argument('--batch', nargs='?', default='001')
    parser.add_argument('--name',  nargs='?', default='0849.4492')
    parser.add_argument('--set',   nargs='?', default='nominal')
    args = parser.parse_args()
    inputargs = {
        'seed'  : (int(args.seed) if args.seed else None),
        'batch' : args.batch,
        'name'  : args.name,
        'set'   : args.set,
    }

with open('locations.json', 'r') as rf:
    config = json.load(rf)

print('Seed for random sampling is: {}'.format(inputargs['seed']))

#-----------------------------------------------------------------------
## Location of event selections config files:
int_config = '../config/int_config.py'
osc_config = '../config/osc_config.py'

## Path to models, scenarios, outdir:
scenarios = config['DATAANA']+'scenarios/'
outdir_   = config['OUTDIR']+'{}-{}-{}_seed{}/'.format(inputargs['batch'], inputargs['name'], inputargs['set'], str(inputargs['seed']))

## Data sets used in this analysis:
sets = {
    # set             location            name
    'nominal'      : ['nominal/',         'nominal'],
    'DOMeff0.90'   : ['sys-DOMeff0.90/',  'DOMeff 0.90'],
    'DOMeff1.10'   : ['sys-DOMeff1.10/',  'DOMeff 1.10'],
    'BIceAbs0.95'  : ['sys-BIceAbs0.95/', 'BIce abs 0.95'],
    'BIceAbs1.05'  : ['sys-BIceAbs1.05/', 'BIce abs 1.05'],
    'BIceSct0.95'  : ['sys-BIceSct0.95/', 'BIce sct 0.95'],
    'BIceSct1.05'  : ['sys-BIceSct1.05/', 'BIce sct 1.05'],
    'HIceP0-1'     : ['sys-HIceP0-1/',    'HIce p0 -1'],
    'HIceP0+1'     : ['sys-HIceP0+1/',    'HIce p0 +1'],
    'OSC-MCv02.02' : ['OSC-MCv02.02/',    'OSC-MC v02.02'],
    'OSC-MCv02.05' : ['OSC-MCv02.05/',    'OSC-MC v02.05'],
    'farsample'    : ['farsample/',       'far sample'],
    'unblinded'    : ['nominal/',         'unblinded'],
}

## Livetime:
yrs9 = 3304     # smaller of the two lifetimes (INT) [days]
#-----------------------------------------------------------------------

## Scenarios are assigned a UUID that corresponds to the mass of the DM
## particle in GeV (e.g.0849.4492). They are grouped into batches (000-143),
## just to keep things a little more organized. In models/ana/ you find
## all batches containing directories for each scenario, which again contain
## sub-directories for each set (nominal, systematic sets, etc.) to save
## the output files in.
## I've picked scenario 0849.4492 of batch 001, with the nominal data set,
## but you can of course pick any scenario with any set. BUT: Systematics
## calculations have not been done for all scenarios, so you might end
## up with nothing to compare with if you pick any set that's not the
## nominal one.

## Here we initialize the scenario:
scen = Model(
    set     = inputargs['set'],
    batch   = inputargs['batch'],
    name    = inputargs['name'],
    allsets = sets,
    loc     = scenarios,
    configs = [int_config, osc_config],
    outdir  = outdir_
)
print(scen.batch, scen.name, scen.set, scen.outdir)

## Create scenario-specific output dir:
if not os.path.exists(outdir_):
    os.mkdir(outdir_)

## The background PDFs are the same for all scenarios, but the signal
## PDFs are not (obviously). Since we use two different event selections,
## Improved Nothern Tracks (INT) and OscNext (OSC) (more on this in the
## wiki), we need two separate background PDFs overall, and two separate
## signal PDFs for each scenario. Those are calculated here:

scen.CreatePDF('INT', 'B')
scen.CreatePDF('INT', 'S')
scen.CreatePDF('OSC', 'B')
scen.CreatePDF('OSC', 'S')


## The next step is the background test statistics. It is calculated by
## creating many test samples (by scrambling the oversampled background
## events) and determine the number of signal events and TS for each of
## the test samples. You can see the definition of the likelihood function
## test statistics in the wiki. This is the first time we use random
## sampling.

TSdist(scen, tests=1000, mu=0, livetime=yrs9, seed=inputargs['seed'])


## The next step is the sensitivity. It is calculated with the TSdist()
## function as well, but this time we inject some number of signal events
## to the test samples. This is done for 5 different numbers of signal
## events (including 0) for which we determine the ratio of test samples
## that yielded TS>0, and the total number of test samples. The five ratios
## are fitted with a saturation function to determine the average median
## upper limit at 90% confidence level.

Sensitivity(scen, livetime=yrs9, seed=inputargs['seed'])
