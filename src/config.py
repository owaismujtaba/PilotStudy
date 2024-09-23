import os
from pathlib import Path

# Functionality
denormalizeData = True
createBIDSFile = True
extractSyllableWordData = False
trainModels = False



# Directories
currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')
dataDir = Path(currDir, 'Data')

nJobs = 4
tmax = 1.5
tmin = -0.5
