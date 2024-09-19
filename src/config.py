import os
from pathlib import Path


createBIDSFile = True
loadData = False

currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData')
bidsDir = Path(currDir, 'BIDS')

