import src.config as config
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd
import pdb
import json
from pathlib import Path
import os
from src.utils import printSectionFooter, printSectionHeader

class NeuralDatasetExtractor:
    """
    A class to handle neural data processing for EEG experiments.

    This class provides functionality to load, process, and extract EEG data
    for words and syllables from BIDS-formatted datasets. It includes methods
    for initializing the dataset, extracting event information, and creating
    epochs for both words and syllables based on the speech type, language element, 
    event type, trial phase, and presentation mode.

    Key functionalities include:
    - Loading and processing raw EEG data
    - Extracting and managing event information
    - Creating and storing word and syllable epochs
    - Handling BIDS file paths and directory structures
    - Managing dictionaries for words, syllables, and their event codes

    Attributes:
        subjectId (str): The ID of the subject.
        sessionId (str): The ID of the session.
        runId (str): The ID of the run.
        taskName (str): The name of the task.
        bidsDir (str): The root directory of the BIDS dataset.
        speechType (str): The type of speech (e.g., 'Overt', 'Covert').
        languageElement (str): The language element type (e.g., 'Word', 'Syllable').
        eventType (str): The type of event (e.g., 'Start', 'End').
        trialPhase (str): The phase of the trial (e.g., 'Stimulus', 'ITI', 'ISI', 'Speech',
        'Fixation', 'Response').
        presentationMode (str): The mode of presentation (e.g., 'Audio', 'Text', 'Picture).
        wordsInExperiment (dict): A dictionary to store words encountered in the experiment.    
        syllablesInExperiment (dict): A dictionary to store syllables encountered in the experiment.
        bidsFilepath (BIDSPath): The BIDS file path for the current dataset.
        rawData (mne.io.Raw): The raw EEG data.
        channels (list): List of channel names in the raw data.
        events (numpy.ndarray): The events extracted from the raw data.
        eventIds (dict): A dictionary mapping event descriptions to their numerical IDs.
        eventIdsReversed (dict): A dictionary mapping numerical event IDs to their descriptions.
        syllablesDict (dict): A dictionary of syllables and their codes.    
        wordsDict (dict): A dictionary of words and their codes.
        wordEpochs (mne.Epochs): Epochs object for word events.
        syllableEpochs (mne.Epochs): Epochs object for syllable events.
    """

    def __init__(self, 
            subjectId='01', sessionId='01', runId='01', 
            taskName='PilotStudy', bidsDir=config.bidsDir, 
            speechType=None, languageElement=None,
            eventType=None, trialPhase=None, presentationMode=None
        ):
        
        printSectionHeader(" Initializing NeuralDatasetExtractor ")
        
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        self.speechType = speechType
        self.languageElement = languageElement
        self.eventType = eventType
        self.trialPhase = trialPhase
        self.presentationMode = presentationMode
        self.wordsInExperiment = {}
        self.syllablesInExperiment = {}
        self.frequencyRange = np.logspace(*np.log10([1, 120]), num=10)

        
        self.bidsFilepath = BIDSPath(
            subject=self.subjectId, session=self.sessionId, run=self.runId,
            task= self.taskName, suffix='eeg',root=self.bidsDir
        )

        printSectionHeader("Loading Data")
        self.rawData = read_raw_bids(self.bidsFilepath)
        self.channels = self.rawData.ch_names
        self.rawData.load_data()
        printSectionFooter("Data Loading Complete")
    
        #self.preprocessData()
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        
        printSectionFooter("✅  Initialization Complete  ✅")
        self.getRealtedEvents()
        self.extractEvents()
        
    def preprocessData(self):
        """
        Apply preprocessing steps to the raw EEG data.

        This method applies the following preprocessing steps:
        1. Notch filter at 50 Hz and 100 Hz to remove power line noise
        2. Bandpass filter between 0.1 Hz and 120 Hz
        3. Independent Component Analysis (ICA) that captures 99% of the variance

        Returns:
            None
        """
        printSectionHeader(" Applying preprocessing steps to the raw EEG data ")
        
        self.rawData.notch_filter([50, 100])
        self.rawData.filter(l_freq=0.1, h_freq=120)

        ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
        rawDataForICA = self.rawData.copy()
        ica.fit(rawDataForICA)
        ica.apply(self.rawData)
        self.rawData.set_eeg_reference(ref_channels=['FCz'])

        printSectionFooter("✅  Preprocessing Completed  ✅")

    def getRealtedEvents(self):
        printSectionHeader('')
        intrestedEvents = []
        for index in range(len(self.events)):
            eventCode = self.events[index][2]
            eventName = self.eventIdsReversed[eventCode]
            if (self._checkSpeechType(eventName, self.speechType) and 
                self._checkLanguageElement(eventName, self.languageElement) and 
                self._checkEventType(eventName, self.eventType) and 
                self._checkTrialPhase(eventName, self.trialPhase) and 
                self._checkPresentationMode(eventName, self.presentationMode)
            ):
                intrestedEvents.append(self.events[index])
        self.intrestedEvents = np.array(intrestedEvents)
    
    def extractEvents(self):
        self.epochsData = mne.Epochs(
            self.rawData, 
            events=self.intrestedEvents, 
            tmin=config.tmin, tmax=config.tmax,
            baseline=(-0.5, 0), 
            picks=self.channels, 
            preload=True,
            verbose=False
        )

    def computeMorletFeatures(self):
        self.morletFeatures = self.epochsData.compute_tfr(
            method='morlet',
            freqs=self.frequencyRange,
            n_cycles = self.frequencyRange/8,
            use_fft=True,
            return_itc=False,
            average=False
        )
    

            
    def _checkSpeechType(self, eventName, speechType):
        """
        Check if the event is in the speech type.

        Args:
            eventName (str): The name of the event to check.
            speechType (str): The type of speech to check against.

        Returns:
            bool: True if the event is in the speech type, False otherwise.
        """
        if speechType == None:
            return True
        if speechType in eventName:
            return True
        return False

    def _checkLanguageElement(self, eventName, languageElement):
        """
        Check if the event is a word.

        Args:
            eventName (str): The name of the event to check.
            languageElement (str): The language element to check against.

        Returns:
            bool: True if the event is a word, False otherwise.
        """
        if languageElement == None:
            return True
        if languageElement in eventName:
            return True
        return False

    def _checkEventType(self, eventName, eventType):
        """
        Check if the event is in the experiment phase.

        Args:
            eventName (str): The name of the event to check.
            eventType (str): The type of event to check against.

        Returns:
            bool: True if the event is in the experiment phase, False otherwise.
        """
        if eventType == None:
            return True
        if eventType in eventName:
            return True
        return False

    def _checkTrialPhase(self, eventName, trialPhase):
        """
        Check if the event is in the event phase.

        Args:
            eventName (str): The name of the event to check.
            trialPhase (str): The trial phase to check against.

        Returns:
            bool: True if the event is in the event phase (i.e., a 'Start' event), False otherwise.
        """
        if trialPhase == None:
            return True
        if trialPhase in eventName:
            return True
        return False
    
    def _checkPresentationMode(self, eventName, presentationMode):
        """
        Check if the event is in the trial phase.

        Args:
            eventName (str): The name of the event to check.
            presentationMode (str): The presentation mode to check against.

        Returns:
            bool: True if the event is in one of the trial phases ('Stimulus', 'ISI', 'Speech', 'ITI', 'Fixation'), False otherwise.
        """
        if presentationMode == None:
            return True
        if presentationMode in eventName:
            return True
        return False
    