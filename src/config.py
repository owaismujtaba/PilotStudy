import os
from pathlib import Path

# Functionality
createBIDSFile = False
loadData = True

# Directories
currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')
dataDir = Path(currDir, 'Data')


tmax = 1.0
tmin = 0.0
