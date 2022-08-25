import numpy as np
import pandas as pd

#--- Data I/O ----------------------------------------------------------
def LoadSample(filename):
    data = np.load(filename, allow_pickle=True)
    sample = pd.DataFrame(data)
    return sample

def SaveSample(sample, outname):
    savesample = sample.to_records(index=False)
    # savesample = sample.to_records()
    # print(savesample)
    np.save(outname, savesample)
