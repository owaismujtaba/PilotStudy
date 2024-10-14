import os
from pathlib import Path
import shutil

# Functionality
createBIDSFile = False
loadData = False 
trainModels = True
visualization = False
extractEpochsData = False


# Modality selction Variable for extracting the events 
TASK_NAME = 'PilotStudy'
SPEECH_TYPE =None
LANGUAGE_ELEMENT = None
EVENT_TYPE ='Experiment'
START_END = 'Start'
TRIAL_PHASE = None
PRESENTATION_MODE=None



EPOCHS=100
BATCHSIZE=32
NJOBS = 1
DEVICE ='GPU'


# Directories
CURRENT_DIR = os.getcwd()
BIDS_DIR = Path(CURRENT_DIR, 'BIDS')
DATA_DIR = Path(CURRENT_DIR, 'Data')
RESULTS_DIR = Path(CURRENT_DIR, 'Results')
IMAGES_DIR = Path(CURRENT_DIR, 'Images')

# For epochs data extraction
T_MAX = 1.5
T_MIN = -0.5
NUM_CLASSES = 5

TERMINAL_WIDTH = shutil.get_terminal_size().columns
