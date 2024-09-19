import os
from pathlib import Path


createBIDSFile = True
loadData = True

currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')

