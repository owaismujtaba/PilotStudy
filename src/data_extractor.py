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
        
        printSectionHeader("🚀 Initializing NeuralDatasetExtractor 🚀")
        
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

        
        self.bidsFilepath = BIDSPath(
            subject=self.subjectId, session=self.sessionId, run=self.runId,
            task= self.taskName, suffix='eeg',root=self.bidsDir
        )

        self.rawData = read_raw_bids(self.bidsFilepath)
        self.channels = self.rawData.ch_names
        self.rawData.load_data()
    
        self.preprocessData()
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {str(value): key for key, value in self.eventIds.items()}
        
        self._getListOfSyllablesAndWords()
        self.extractEegDataForWords()
        self.extractEegDataForSyllables()
        
        printSectionFooter("✅  Initialization Complete  ✅")
        
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
        printSectionHeader("🔧 Applying preprocessing steps to the raw EEG data 🔧")
        
        self.rawData.notch_filter([50, 100])
        self.rawData.filter(l_freq=0.1, h_freq=None)

        ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
        ica.fit(self.rawData)
        self.rawData = ica.apply(self.rawData)
        self.rawData.set_eeg_reference(ref_channels=['FCz'])

        printSectionFooter("✅  Preprocessing Completed  ✅")

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
        printSectionHeader("📜 Getting list of syllables and words 📜")
        
        syllables = {}
        words = {}
        syllableIndex = 0
        wordIndex = 0
        for key, eventDetails in self.eventIdsReversed.items():
                if 'Syllable' in eventDetails:
                    syllable = eventDetails.split('_')[-1]
                    if syllable not in syllables:
                        syllables[syllable] = syllableIndex
                        syllableIndex += 1
                elif 'Word' in eventDetails:
                    word = eventDetails.split('_')[-1]
                    if word not in words:
                        words[word] = wordIndex
                        wordIndex += 1
                
                
        self.syllablesDict = syllables
        self.wordsDict = words

        printSectionFooter("✅  Syllables and Words Extracted  ✅")

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
        printSectionHeader("🧠 Extracting EEG data for words 🧠")
        
        codes, eventTimings = [], []
        for event in self.events:
            eventName = self.eventIdsReversed[str(event[2])]
            if (self._checkSpeechType(eventName, self.speechType) and 
                self._checkLanguageElement(eventName, self.languageElement) and 
                self._checkEventType(eventName, self.eventType) and 
                self._checkTrialPhase(eventName, self.trialPhase) and 
                self._checkPresentationMode(eventName, self.presentationMode)
            ): 
                word = eventName.split('_')[-1]
                code = self._getCodeForWord(word)
                if code is not None:
                    codes.append(code)
                    eventTimings.append(event[0])
        wordsEvents = np.array([[timing, 0, code] for timing, code in zip(eventTimings, codes)])
        
        self.wordEpochs = mne.Epochs(self.rawData, wordsEvents, 
                                     event_id=self.wordsInExperiment, 
                                     tmin=config.tmin, tmax=config.tmax,
                                     baseline=(-0.5, 0), 
                                     picks=self.channels)

        printSectionFooter("✅  Word EEG Data Extraction Complete  ✅")

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
        printSectionHeader("🧠 Extracting EEG data for syllables 🧠")
        
        codes, eventTimings = [], []

        for event in self.events:
            eventName = self.eventIdsReversed[str(event[2])]
            if (self._checkSpeechType(eventName, self.speechType) and   
                self._checkLanguageElement(eventName, self.languageElement) and 
                self._checkEventType(eventName, self.eventType) and     
                self._checkTrialPhase(eventName, self.trialPhase) and 
                self._checkPresentationMode(eventName, self.presentationMode)
            ):

                syllable = eventName.split('_')[-1] 
                code = self._getCodeForSyllable(syllable)
                if code is not None:
                    codes.append(code)
                    eventTimings.append(event[0])


        syllableEvents = [[timing, 0, code] for timing, code in zip(eventTimings, codes)]
        syllableEvents = np.array(syllableEvents)   


        self.syllableEpochs = mne.Epochs(
            self.rawData, syllableEvents, 
            event_id=self.syllablesInExperiment, 
            tmin=config.tmin, tmax=config.tmax
        )

        printSectionFooter("✅  Syllable EEG Data Extraction Complete  ✅")
        
    

class WordSyllableDataExtractor:
    """
    A class to extract and save word and syllable data from BIDS-formatted EEG data.

    This class initializes a NeuralDatasetExtractor object which creates data Epochs 
    for words and syllabels. This class provides methods to extract and save word and
    syllable data for a given subject, session, and specific parameters.


    Attributes:
        bidsDir (str): The root directory of the BIDS dataset.
        subjectId (str): The ID of the subject.
        sessionId (str): The ID of the session.
        runId (str): The ID of the run.
        taskName (str): The name of the task.
        speechType (str): The type of speech (e.g., 'Overt', 'Covert').
        languageElement (str): The language element type (e.g., 'Word', 'Syllable').
        eventType (str): The type of event (default is 'Start').
        trialPhase (str): The phase of the trial.
        presentationMode (str): The mode of presentation.
        destionationDataDir (Path): The directory to save the extracted data.
        neurlDatasetObject (NeuralDatasetExtractor): The NeuralDatasetExtractor object.

    Methods:
        extractWordData(): Extracts and saves word data as NumPy arrays.
        extractSyllableData(): Extracts and saves syllable data as NumPy arrays.
    """
    def __init__(self, 
            bidsDir=config.bidsDir, subjectId='01', sessionId='01', 
            runId='01', taskName='PilotStudy', 
            speechType=None, languageElement=None, eventType='Start', trialPhase=None, presentationMode=None
        ):
        
        printSectionHeader("🚀 Initializing WordSyllableDatasetExtractor 🚀")
        
        print(f"bidsDir={bidsDir}, subjectId={subjectId}, sessionId={sessionId}, runId={runId}, taskName={taskName}")
        self.bidsDir = bidsDir
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId      
        self.taskName = taskName
        self.speechType = speechType
        self.languageElement = languageElement
        self.eventType = eventType
        self.trialPhase = trialPhase
        self.presentationMode = presentationMode
        self.destionationDataDir = Path(
            config.dataDir, 'sub-'+self.subjectId, 'ses-'+self.sessionId,
            str(self.speechType)+'_'+str(self.languageElement)+'_'+str(self.eventType)
            +'_'+str(self.trialPhase)+'_'+str(self.presentationMode)
        )
        self.neurlDatasetObject = NeuralDatasetExtractor(
            subjectId=self.subjectId, 
            sessionId=self.sessionId, 
            runId=self.runId, 
            taskName=self.taskName, 
            bidsDir=self.bidsDir,
            speechType=self.speechType,
            languageElement=self.languageElement,
            eventType=self.eventType,
            trialPhase=self.trialPhase,
            presentationMode=self.presentationMode
        )

        self.extractWordData()
        self.extractSyllableData()

        printSectionFooter("✅  WordSyllableDataExtractor Initialization Complete  ✅")

    def extractWordData(self):
        printSectionHeader(f"📝 Extracting word data for subject {self.subjectId}, session {self.sessionId} 📝")
        
        self.wordDataEpochs = self.neurlDatasetObject.wordEpochs
        wordDir = Path(self.destionationDataDir, 'Words')
        print(f"💾 Saving word data to {wordDir}")
        os.makedirs(wordDir, exist_ok=True)
        for word, code in self.neurlDatasetObject.wordsInExperiment.items():
            wordData = self.wordDataEpochs[word].get_data()
            self.destinationPath = Path(wordDir, word+'.npy')
            np.save(self.destinationPath, wordData)

        printSectionFooter("✅  Word Data Extraction Complete  ✅")

    def extractSyllableData(self):
        printSectionHeader(f"📝 Extracting syllable data for subject {self.subjectId}, session {self.sessionId} 📝")
        
        syllableDir = Path(self.destionationDataDir, 'Syllables')
        print(f"💾 Saving syllable data to {syllableDir}")
        os.makedirs(syllableDir, exist_ok=True)
        self.syllableDataEpochs = self.neurlDatasetObject.syllableEpochs
        for syllable, code in self.neurlDatasetObject.syllablesInExperiment.items():
            syllableData = self.syllableDataEpochs[syllable].get_data()
            self.destinationPath = Path(syllableDir, syllable+'.npy')
            np.save(self.destinationPath, syllableData)

        printSectionFooter("✅  Syllable Data Extraction Complete  ✅")


def extractWordSyllableDataForAllSubjects(
        speechType=None, languageElement=None, eventType='Start', 
        trialPhase=None, presentationMode=None):
    """
    Extract word and syllable data for all subjects in the BIDS directory.

    This function iterates through all subjects and sessions in the BIDS directory
    and uses the WordSyllableDataExtractor class to extract and save the data.

    Args:
        speechType (str, optional): The type of speech to extract (e.g., 'Overt', 'Covert'). Defaults to None.
        languageElement (str, optional): The language element to extract (e.g., 'Word', 'Syllable'). Defaults to None.
        eventType (str, optional): The type of event to extract. Defaults to 'Start'.
        trialPhase (str, optional): The trial phase to extract. Defaults to None.
        presentationMode (str, optional): The presentation mode to extract. Defaults to None.

    Note: This function assumes a specific BIDS directory structure and naming convention.
    """
    rootDir = Path(config.bidsDir)

    for subject in [dir for dir in os.listdir(rootDir) if os.path.isdir(Path(rootDir, dir))]:
        subjectPath = Path(rootDir, subject)
        for session in [dir for dir in os.listdir(subjectPath) if os.path.isdir(Path(subjectPath, dir))]:
            subjectId = subject.split('-')[-1]
            sessionId = session.split('-')[-1]
            runId = '01'  # Assuming a fixed run ID
            taskName = 'PilotStudy'  # Assuming a fixed task name

            printSectionHeader(f"📊 Extracting data for subject {subjectId}, session {sessionId} 📊")
            print(f"Parameters:".center(60))
            print(f"Speech Type: {speechType}".center(60))
            print(f"Language Element: {languageElement}".center(60))
            print(f"Event Type: {eventType}".center(60))
            print(f"Trial Phase: {trialPhase}".center(60))
            print(f"Presentation Mode: {presentationMode}".center(60))
            print("=" * 60 + "\n")
            
            wordSyllableDataExtractor = WordSyllableDataExtractor(
                bidsDir=config.bidsDir,
                subjectId=subjectId, 
                sessionId=sessionId, 
                runId=runId, 
                taskName=taskName,
                speechType=speechType,
                languageElement=languageElement,
                eventType=eventType,
                trialPhase=trialPhase,
                presentationMode=presentationMode
            )

