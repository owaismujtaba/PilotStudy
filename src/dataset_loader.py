import src.config as config
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd
import pdb
from pathlib import Path
import os

class NeuralDataset:
    """
    A class to handle neural data processing for EEG experiments.

    This class provides functionality to load, process, and extract EEG data
    for words and syllables from BIDS-formatted datasets. It includes methods
    for initializing the dataset, extracting event information, and creating
    epochs for both words and syllables.

    Attributes:
        subjectId (str): The ID of the subject.
        sessionId (str): The ID of the session.
        runId (str): The ID of the run.
        taskName (str): The name of the task.
        bidsDir (str): The root directory of the BIDS dataset.
        wordsInExperiment (dict): A dictionary to store words encountered in the experiment.
        bidsFilepath (BIDSPath): The BIDS file path for the current dataset.
        rawData (mne.io.Raw): The raw EEG data.
        events (numpy.ndarray): The events extracted from the raw data.
        eventIds (dict): A dictionary mapping event descriptions to their numerical IDs.
        eventIdsReversed (dict): A dictionary mapping numerical event IDs to their descriptions.
        syllablesDict (dict): A dictionary of syllables and their indices.
        wordsDict (dict): A dictionary of words and their indices.
        wordEpochs (mne.Epochs): Epochs object for word events.
        syllableEpochs (mne.Epochs): Epochs object for syllable events.
    """

    def __init__(self, subjectId='01', sessionId='01', runId='01', taskName='PilotStudy', bidsDir=config.bidsDir) -> None:
        """
        Initialize the NeuralDataset class.

        This method sets up the basic attributes of the dataset, loads the raw data,
        extracts events, and initializes the processing of words and syllables.

        Args:
            subjectId (str): The ID of the subject.
            sessionId (str): The ID of the session.
            runId (str): The ID of the run.
            taskName (str): The name of the task.
            bidsDir (str): The root directory of the BIDS dataset.

        Returns:
            None
        """
        print("Initializing NeuralDataset")
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        self.wordsInExperiment = {}
        self.syllablesInExperiment = {}
        
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
        
        self._getListOfSyllablesAndWords()

        self.extractEegDataForWords()
        self.extractEegDataForSyllables()

    def _getListOfSyllablesAndWords(self):
        """
        Get the list of syllables and words from the event IDs.

        This method processes the event IDs to create dictionaries of unique
        syllables and words encountered in the experiment, assigning each a
        unique index.

        The method populates two instance attributes:
        - syllablesDict: A dictionary where keys are syllables and values are their indices.
        - wordsDict: A dictionary where keys are words and values are their indices.

        Note: This method skips 'silence' events when processing words.
        """
        print("Getting list of syllables and words")
        syllables = {}
        words = {}
        syllableIndex = 0
        wordIndex = 0
        for key, eventDetails in self.eventIdsReversed.items():
            if 'Experiment' in eventDetails and 'Start' in eventDetails:
                if 'Syllable' in eventDetails:
                    syllable = eventDetails.split(',')[-1]
                    if syllable not in syllables:
                        syllables[syllable] = syllableIndex
                        syllableIndex += 1
                elif 'Word' in eventDetails:
                    word = eventDetails.split(',')[-1]
                    if word == 'silence':
                        continue
                    if word not in words:
                        words[word] = wordIndex
                        wordIndex += 1
                
                
        self.syllablesDict = syllables
        self.wordsDict = words

    def _checkExperimentPhase(self, eventName):
        """
        Check if the event is in the experiment phase.

        Args:
            eventName (str): The name of the event to check.

        Returns:
            bool: True if the event is in the experiment phase, False otherwise.
        """
        if 'Experiment' in eventName:
            return True
        return False
    
    def _checkEventPhase(self, eventName):
        """
        Check if the event is in the event phase.

        Args:
            eventName (str): The name of the event to check.

        Returns:
            bool: True if the event is in the event phase (i.e., a 'Start' event), False otherwise.
        """
        if 'Start' in eventName:
            return True
        return False
    
    def _checkTrialPhase(self, eventName):
        """
        Check if the event is in the trial phase.

        Args:
            eventName (str): The name of the event to check.

        Returns:
            bool: True if the event is in the trial phase (i.e., a 'Speech' event), False otherwise.
        """
        if 'Speech' in eventName:
            return True
        return False
      
    def _checkWordType(self, eventName):
        """
        Check if the event is a word.

        Args:
            eventName (str): The name of the event to check.

        Returns:
            bool: True if the event is a word, False otherwise.
        """
        if 'Word' in eventName:
            return True
        return False

    def extractEegDataForWords(self):
        """
        Extract EEG data for words.

        This method processes the events to extract EEG data specifically for word events.
        It creates an Epochs object (self.wordEpochs) containing the EEG data
        time-locked to the onset of each word event.

        The method filters events based on experiment phase, trial phase, event phase,
        and word type. It then creates an array of word events and uses it to epoch
        the raw EEG data.

        The resulting Epochs object is stored in self.wordEpochs.
        """
        print("Extracting EEG data for words")
        codes, eventTimings = [], []
        for event in self.events:
            eventName = self.eventIdsReversed[str(event[2])]
            if (self._checkExperimentPhase(eventName) and 
                self._checkTrialPhase(eventName) and 
                self._checkEventPhase(eventName) and 
                self._checkWordType(eventName)):
                
                word = eventName.split(',')[-1]
                code = self._getCodeForWord(word)
                if code is not None:
                    codes.append(code)
                    eventTimings.append(event[0])
        
        wordsEvents = np.array([[timing, 0, code] for timing, code in zip(eventTimings, codes)])
        
        self.wordEpochs = mne.Epochs(self.rawData, wordsEvents, 
                                     event_id=self.wordsInExperiment, 
                                     tmin=config.tmin, tmax=config.tmax, 
                                     baseline=None, 
                                     picks=self.rawData.ch_names)

    def extractEegDataForSyllables(self):
        """
        Extract EEG data for syllables.

        This method processes the events to extract EEG data specifically for syllable events.
        It creates an Epochs object (self.syllableEpochs) containing the EEG data
        time-locked to the onset of each syllable event.

        The method filters events based on experiment phase, trial phase, event phase,
        and ensures they are not word events. It then creates an array of syllable events 
        and uses it to epoch the raw EEG data.

        The resulting Epochs object is stored in self.syllableEpochs.
        """
        print("Extracting EEG data for syllables")
        codes, eventTimings = [], []
        for event in self.events:
            eventName = self.eventIdsReversed[str(event[2])]
            if (self._checkExperimentPhase(eventName) and 
                self._checkTrialPhase(eventName) and 
                self._checkEventPhase(eventName) and 
                not self._checkWordType(eventName)):
                
                syllable = eventName.split(',')[-1]
                code = self._getCodeForSyllable(syllable)
                if code is not None:
                    codes.append(code)
                    eventTimings.append(event[0])
        syllablesEvents = np.array([[timing, 0, code] for timing, code in zip(eventTimings, codes)])
        self.syllableEpochs = mne.Epochs(self.rawData, syllablesEvents, 
                                         event_id=self.syllablesInExperiment, 
                                         tmin=config.tmin, tmax=config.tmax, 
                                         baseline=None, 
                                         picks=self.rawData.ch_names)
        
    def _getCodeForWord(self, word):
        """
        Get the code for a word.

        This method retrieves the numerical code for a given word from the wordsDict.
        If the word is found, it's added to the wordsInExperiment dictionary.

        Args:
            word (str): The word to get the code for.

        Returns:
            int: The numerical code for the word if it exists in wordsDict, None otherwise.
        """
        if word in self.wordsDict:
            self.wordsInExperiment[word] = self.wordsDict[word]
            return self.wordsDict[word]
        
        return None

    def _getCodeForSyllable(self, syllable):
        """
        Get the code for a syllable.

        This method retrieves the numerical code for a given syllable from the syllablesDict.
        If the syllable is found, it's added to the syllablesInExperiment dictionary.

        Args:
            syllable (str): The syllable to get the code for.

        Returns:
            int: The numerical code for the syllable if it exists in syllablesDict, None otherwise.
        """
        if syllable in self.syllablesDict:
            self.syllablesInExperiment[syllable] = self.syllablesDict[syllable]
            return self.syllablesDict[syllable]
        return None


class WordSyllableDatasetExtractor:

    def __init__(self, bidsDir=config.bidsDir, subjectId='01', sessionId='01', runId='01', taskName='PilotStudy'):
        
        print(f"Initializing WordSyllableDatasetExtractor with: ")
        print(f"bidsDir={bidsDir}, subjectId={subjectId}, sessionId={sessionId}, runId={runId}, taskName={taskName}")
        self.bidsDir = bidsDir
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId      
        self.taskName = taskName
        self.destionationDataDir = Path(config.dataDir, 'sub-'+self.subjectId, 'ses-'+self.sessionId, 'eeg')
        self.neurlDatasetObject = NeuralDataset(
            subjectId=self.subjectId, 
            sessionId=self.sessionId, 
            runId=self.runId, 
            taskName=self.taskName, 
            bidsDir=self.bidsDir
        )
        self.extractWordData()
        self.extractSyllableData()

    def extractWordData(self):
        print(f"Extracting word data for subject {self.subjectId}, session {self.sessionId}")
        self.wordsEpochData = self.neurlDatasetObject.wordEpochs
        wordDir = Path(self.destionationDataDir, 'Words')
        print(f"Saving word data to {wordDir}")
        os.makedirs(wordDir, exist_ok=True)
        for word, code in self.neurlDatasetObject.wordsInExperiment.items():
            wordData = self.wordsEpochData[word].get_data()
            self.destinationPath = Path(wordDir, word+'.npy')
            np.save(self.destinationPath, wordData)

    def extractSyllableData(self):
        print(f"Extracting syllable data for subject {self.subjectId}, session {self.sessionId}")
        syllableDir = Path(self.destionationDataDir, 'Syllables')
        print(f"Saving syllable data to {syllableDir}")
        os.makedirs(syllableDir, exist_ok=True)
        self.syllableEpochData = self.neurlDatasetObject.syllableEpochs
        for syllable, code in self.neurlDatasetObject.syllablesInExperiment.items():
            syllableData = self.syllableEpochData[syllable].get_data()
            self.destinationPath = Path(syllableDir, syllable+'.npy')
            np.save(self.destinationPath, syllableData)
        

def extractWordSyllableDataForAllSubjects():
    rootDir = Path(config.bidsDir)

    for subject in [dir for dir in os.listdir(rootDir) if os.path.isdir(Path(rootDir, dir))]:
        subjectPath = Path(rootDir, subject)
        for session in  [dir for dir in os.listdir(subjectPath) if os.path.isdir(Path(subjectPath, dir))]:
            subjectId = subject.split('-')[-1]
            sessionId = session.split('-')[-1]
            runId = '01'
            taskName = 'PilotStudy'
            print(f"Extracting data for subject {subjectId}, session {sessionId}")
            obj = WordSyllableDatasetExtractor(
                subjectId=subjectId, 
                sessionId=sessionId, 
                runId=runId, 
                taskName=taskName
            )
