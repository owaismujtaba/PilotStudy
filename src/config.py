import os
from pathlib import Path


createBIDSFile = True

currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')

