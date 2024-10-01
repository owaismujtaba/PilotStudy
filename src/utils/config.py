import os
from pathlib import Path
import shutil

# Functionality
createBIDSFile = False
loadData = False 
trainModels = False
visualization = True


# Directories
currDir = os.getcwd()
rawDataDir = Path(currDir, 'RawData', 'EEG')
bidsDir = Path(currDir, 'BIDS')
dataDir = Path(currDir, 'Data')
resultsDir = Path(currDir, 'Results')
imagesDir = Path(currDir, 'Images')
device='GPU'
nJobs = 15
tmax = 1.5
tmin = -0.5
nClasses = 5

terminalWidth = shutil.get_terminal_size().columns
