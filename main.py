from src.data_utils import XDFData
import src.config as config


filepath='RawData/EEG/sub-P014/ses-S001/eeg/sub-P014_ses-S001_task-Default_run-001_eeg.xdf'
subjectId = '14'
sessionId = '01'
runId = '01'
print(filepath)
print(f'Subject Id:: {subjectId} Session Id:: {sessionId}')
if config.createBIDSFile:

    data = XDFData(
        filepath=filepath,
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId
    )