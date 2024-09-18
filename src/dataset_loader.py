import src.config as config
import mne
from mne_bids import BIDSPath, read_raw_bids

import pdb

class SyllableDataset:
    def __init__(self, subjectId='01', sessionId='01', runId='01', taskName='PilotStudy', bidsDir=config.bidsDir) -> None:
        
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        
        
        self.bidsFilepath = BIDSPath(
            subject=self.subjectId,
            session=self.sessionId,
            run=self.runId,
            task=taskName,
            suffix='eeg',
            root=self.bidsDir
        )

        self.rawData = read_raw_bids(self.bidsFilepath)
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {str(value): key for key, value in self.eventIds.items()}
        self._getSyllablesAndWords()
    def SyllabelExtractor(self):
        
        
        for key, item in self.eventIdsReversed.items():
            pass

    def _getSyllablesAndWords(self):
        syllabels = []
        words = []
        for key, item in self.eventIdsReversed.items():
            if 'End' in item or 'Practice' in item:
                continue
            if 'Word' in item:
                word = item.split(',')[-1]
                words.append(word)
            else:
                syllable = item.split(',')[-1]
                syllabels.append(syllable)
        self.words = words
        self.syllables = syllabels

    def _checkSyllableAndWOrd(self, event):
        # returns True for Syllable and False for Word
        if 'Syllable' in event and 'Experiment':
            return True
        else:
            return False
            