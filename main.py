import pdb
from src.bids_dataset import XDFData
import src.config as config
from src.dataset_loader import SyllableDataset

filepath='RawData\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'
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

