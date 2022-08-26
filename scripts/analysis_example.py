#!/usr/bin/env python3
import os
import shutil
import sys
import time
import pickle
import numpy as np
from numpy import nan
import pandas as pd
from datetime import datetime

from solarinelastic import Model
from solarinelastic import Selection
from solarinelastic.frequentist import *


##--- Analysis configuration -------------------------------------------
## Location of event selections config files:
int_config = '/data/user/rbusse/analysis/selections/int/int_config.py'
osc_config = '/data/user/rbusse/analysis/selections/oscnext/osc_config.py'

## Path to models, scenarios:
scenariospath  = '/data/user/rbusse/analysis/models/ana/'

## Data sets used in this analysis:
sets = {
    # set             location            name
    'nominal'     : ['nominal/',         'nominal'],
    'DOMeff0.90'  : ['sys-DOMeff0.90/',  'DOMeff 0.90'],
    'DOMeff1.10'  : ['sys-DOMeff1.10/',  'DOMeff 1.10'],
    'BIceAbs0.95' : ['sys-BIceAbs0.95/', 'BIce abs 0.95'],
    'BIceAbs1.05' : ['sys-BIceAbs1.05/', 'BIce abs 1.05'],
    'BIceSct0.95' : ['sys-BIceSct0.95/', 'BIce sct 0.95'],
    'BIceSct1.05' : ['sys-BIceSct1.05/', 'BIce sct 1.05'],
    'HIceP0-1'    : ['sys-HIceP0-1/',    'HIce p0 -1'],
    'HIceP0+1'    : ['sys-HIceP0+1/',    'HIce p0 +1'],
}

## Livetime:
yrs9 = 3304     # smaller of the two lifetimes (INT)
##----------------------------------------------------------------------

# set = sys.argv[3]
set = 'nominal'

# batch = str(sys.argv[1])
# name  = str(sys.argv[2])
batch, name = '001', '0849.4492'

## Initialize scenario:
scen = Model(
    set, batch, name,
    allsets = sets,
    loc=scenariospath,
    configs=[int_config, osc_config],
    m=8
)
print('Scenario {} {} loaded succesfully.'.format(scen.batch, scen.name))


##--- Here's a bunch of stuff you can do: ------------------------------

# scen.LoadResults()
# scen.UpdateResults()
# Sensitivity(scen, yrs9)
# MRF(scen, yrs9)

# scen.CreateBPDF('INT')
# scen.FineGridEvaluation('INT', 'B')
# scen.CreateSPDF('INT')
# scen.FineGridEvaluation('INT', 'S')
# scen.INT.DetermineLivetime(yrs9)
# scen.INT.OversampleSignal(set, div=100)

# scen.CreateBPDF('OSC')
# scen.FineGridEvaluation('OSC', 'B')
# scen.CreateSPDF('OSC')
# scen.FineGridEvaluation('OSC', 'S')
# scen.OSC.DetermineLivetime(yrs9)
# scen.OSC.OversampleSignal(set, div=100)


## Determine fractions:
# N = scen.INT.eventsper9years + scen.OSC.eventsper9years
# print(scen.INT.eventsper9years/N)
# print(scen.OSC.eventsper9years/N)
