import pdb
from src.bids_dataset import XDFData
import src.config as config
from src.data_extractor import WordSyllableDataExtractor, extractWordSyllableDataForAllSubjects
from src.dataset_loader import VowelDataset
filepath='RawData/EEG/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

subjectId = '01'
sessionId = '01'
runId = '01'
filepath='RawData/EEG/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

print(filepath)
print(f'Subject Id:: {subjectId} Session Id:: {sessionId}')


if config.extracctVowelData:
    vowelDatasetExtractor = VowelDataset(
        rootDir=config.dataDir,
        subjectId=subjectId,
        sessionId=sessionId,
        speechType='Silent',
        languageElement='Experiment',
        eventType='Start',
        trialPhase=None,
        presentationMode='Speech'
    )
    data, labels = vowelDatasetExtractor.vowelData
    print(data.shape, labels.shape)



if config.extractSyllableWordData:
    extractWordSyllableDataForAllSubjects(
        speechType='Silent',
        languageElement='Experiment',
        eventType='Start',
        trialPhase=None,
        presentationMode='Speech'
    )

if config.createBIDSFile:
    data = XDFData(
        filepath=filepath,
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId
    )


