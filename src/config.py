import os
from pathlib import Path

# Functionality
denormalizeData = True
createBIDSFile = False
extractSyllableWordData = False
extracctVowelData = True



# Directories
currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')
dataDir = Path(currDir, 'Data')

nJobs = 4
tmax = 1.5
tmin = -0.5
