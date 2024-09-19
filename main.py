import pdb
from src.bids_dataset import XDFData
import src.config as config
from src.dataset_loader import SyllableDataset

<<<<<<< HEAD
filepath='RawData/EEG/sub-P014/ses-S001/eeg/sub-P014_ses-S001_task-Default_run-001_eeg.xdf'
=======
filepath='RawData\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
>>>>>>> c795abf7c0b1f27be755a6eacc7ee7e5ba9cd311
subjectId = '01'
sessionId = '01'
runId = '01'
print(filepath)
print(f'Subject Id:: {subjectId} Session Id:: {sessionId}')


if config.loadData:
    obj = SyllableDataset()

    pdb.set_trace()
if config.createBIDSFile:

    data = XDFData(
        filepath=filepath,
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId
    )

