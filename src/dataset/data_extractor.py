import os
import mne
import time
from datetime import timedelta
import numpy as np
from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids
from colorama import Fore, Style, init

import src.utils.config as config
from src.utils.utils import getFolderAndDestination, checkIfEpochsFileExist
from src.utils.utils import printSectionFooter, printSectionHeader

import pdb



init(autoreset=True)


class NeuralDatasetExtractor:
    """
    A class for processing neural data from EEG experiments.
   
    This class provides methods to load, process, and extract EEG data
    from BIDS-formatted datasets. It includes functionality for initializing
    the dataset, extracting event information, and creating epochs for words
    and syllables based on various parameters such as speech type, 
    language element, event type, trail phase and presentation mode
    

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
        bidsFilepath (BIDSPath): The BIDS file path for the current dataset.
        rawData (mne.io.Raw): The raw EEG data.
        channels (list): List of channel names in the raw data.
        events (numpy.ndarray): The events extracted from the raw data.
        eventIds (dict): A dictionary mapping event descriptions to their numerical IDs.
        eventIdsReversed (dict): A dictionary mapping numerical event IDs to their descriptions.
        epochsData (mne.Epochs): Epochs object for the filtered events.
    """

    def __init__(self, 
            subjectId=None, sessionId=None, 
            runId='01', taskName='PilotStudy', 
            bidsDir=config.BIDS_DIR, 
            speechType=config.SPEECH_TYPE, 
            startEnd=config.START_END,
            languageElement=config.LANGUAGE_ELEMENT,
            eventType=config.EVENT_TYPE, trialPhase=None, 
            presentationMode=config.PRESENTATION_MODE
        ):
        
        printSectionHeader(f"{Fore.CYAN} Initializing NeuralDatasetExtractor {Style.RESET_ALL}")
        
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        self.speechType = speechType
        self.languageElement = languageElement
        self.startEnd=startEnd
        self.eventType = eventType
        self.trialPhase = trialPhase
        self.presentationMode = presentationMode
        self.frequencyRange = np.logspace(*np.log10([1, 120]), num=10)
        self.bidsFilepath = BIDSPath(
            subject=self.subjectId, session=self.sessionId, run=self.runId,
            task=self.taskName, suffix='eeg', root=self.bidsDir
        )

        self.displayInfo()

        printSectionFooter(f"{Fore.GREEN} NeuralDatasetExtractor Initialization Complete {Style.RESET_ALL}")
       
        printSectionHeader(f"{Fore.YELLOW} Loading Raw Data {Style.RESET_ALL}")
        self.rawData = read_raw_bids(self.bidsFilepath)
        self.channels = self.rawData.ch_names
        self.rawData.load_data()
        printSectionFooter(f"{Fore.GREEN}✅ Data Loading Complete ✅{Style.RESET_ALL}")
        
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        
        self.preprocessData()
        self.getRealtedEvents()
        self.extractEvents()
        self.saveEpochedData()

    def displayInfo(self):
        """
        Display information about the subject, session, task, and run.

        This method prints out the details of the subject, session, task, and run
        in a clear and visually appealing format.

        Returns:
            None
        """
        color = Fore.MAGENTA
        printSectionHeader("ℹ️ Subject, Session, Task, and Run Information ℹ️")
        print(f"{color}\033[1m\033[4m🧑‍🔬 Subject ID:            {self.subjectId}".ljust(60))  
        print(f"{color}\033[1m📅 Session ID:            {self.sessionId}".ljust(60))   
        print(f"{color}\033[1m🏃‍♂️ Run ID:            {self.runId}".ljust(60))         
        print(f"{color}\033[1m📝 Task Name:             {self.taskName}".ljust(60))    
        print(f"{color}\033[1m🔊 Speech Type:           {self.speechType}".ljust(60)) 
        print(f"{color}\033[1m🔤 Language Element:          {self.languageElement}".ljust(60)) 
        print(f"{color}\033[1m📊 Event Type:            {self.eventType}".ljust(60)) 
        print(f"{color}\033[1m⏳ Start/End:             {self.startEnd}".ljust(60))   
        print(f"{color}\033[1m⏳ Trial Phase:           {self.trialPhase}".ljust(60))   
        print(f"{color}\033[1m🖥️ Presentation Mode:             {self.presentationMode}".ljust(60)) 
        print(f"{color}\033[1m📂 BIDS Directory: {self.bidsDir}{Style.RESET_ALL}".ljust(60))  
    
    def preprocessData(self):
        """
        Apply preprocessing steps to the raw EEG data.

        This method applies the following preprocessing steps:
        1. Notch filter at 50 Hz and 100 Hz to remove power line noise.
        2. Bandpass filter between 0.1 Hz and 120 Hz.
        3. Independent Component Analysis (ICA) to capture 99% of the variance.
        1. Notch filter at 50 Hz and 100 Hz to remove power line noise.
        2. Bandpass filter between 0.1 Hz and 120 Hz.
        3. Independent Component Analysis (ICA) to capture 99% of the variance.

        Returns:
            None
        """
        printSectionHeader(f"{Fore.MAGENTA}粒 Applying Preprocessing Steps 粒{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}1️⃣ Applying notch filter...{Style.RESET_ALL}")
        self.rawData.notch_filter([50, 100])
        
        print(f"{Fore.YELLOW}2️⃣ Applying bandpass filter...{Style.RESET_ALL}")
        self.rawData.filter(l_freq=0.1, h_freq=120, n_jobs=config.NJOBS)

        '''
        print(f"{Fore.YELLOW}3️⃣ Performing ICA...{Style.RESET_ALL}") # Regression Filtering Remove ICA 
        ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
        rawDataForICA = self.rawData.copy()
        ica.fit(rawDataForICA)
        ica.apply(self.rawData)
        '''
        print(f"{Fore.YELLOW}4️⃣ Setting EEG reference to FCz{Style.RESET_ALL}")
        self.rawData.set_eeg_reference(ref_channels=['FCz'])

        printSectionFooter(f"{Fore.GREEN}✨ Preprocessing Completed ✨{Style.RESET_ALL}")

    def getRealtedEvents(self):
        """
        Extract events of interest based on the specified parameters.

        This method filters the events based on the speech type, language element,
        event type, trial phase, and presentation mode specified during initialization.

        Returns:
            None
        """
        printSectionHeader(f" Extracting Related Events ")
        intrestedEvents = []
        for index in range(len(self.events)):
            eventCode = self.events[index][2]
            eventName = self.eventIdsReversed[eventCode]
            if all(self._checkEventProperty(eventName, parm) for parm in [
                self.speechType, self.languageElement, self.eventType, 
                self.startEnd, self.trialPhase, self.presentationMode
            ]):
                intrestedEvents.append(self.events[index])
        self.intrestedEvents = np.array(intrestedEvents)
        print(f"{Fore.MAGENTA} Found {len(self.intrestedEvents)} events of interest{Style.RESET_ALL}".center(60))
        self.displayInfo()
        printSectionFooter(f"✅ Event Extraction Complete ✅")
    
    def extractEvents(self):
        """
        Create epochs from the filtered events.

        This method creates epochs from the events of interest using the specified
        time window and baseline correction.

        Returns:
            None
        """
        printSectionHeader(" Creating Epochs ")
        self.epochsData = mne.Epochs(
            self.rawData, 
            events=self.intrestedEvents, 
            tmin=config.T_MIN, tmax=config.T_MAX,
            baseline=(config.T_MIN, 0), 
            picks=self.channels, 
            preload=True,
            verbose=False
        )

        self.epochsData.event_id = {self.eventIdsReversed[int(key)]: int(key) for key,_ in self.epochsData.event_id.items() }
        print(f"{Fore.GREEN} Created {len(self.epochsData)} epochs{Style.RESET_ALL}".center(60))
        printSectionFooter("✅ Epoch Creation Complete ✅")

    def _checkEventProperty(self, eventName, propertyValue):
        if propertyValue is None:
            return True
        return propertyValue in eventName
    
    def saveEpochedData(self):
        """
        Save the epoched data to a file.

        This method saves the epoched data (self.epochsData) to a file in the specified output directory.
        The file will be in the .fif format, which is the standard format for MNE-Python objects.

        Args:
            output_dir (str): The directory where the epoched data should be saved.

        Returns:
            None
        """
        
        printSectionHeader(" Saving Epoched Data ")

        filename, folder, destinationDir = getFolderAndDestination(
            self.subjectId, self.sessionId, self.taskName,
            self.runId, self.speechType, self.languageElement,
            self.eventType, self.startEnd, self.trialPhase,
            self.presentationMode
        )
        Path(destinationDir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(destinationDir) / filename
        #self.epochsData.apply_baseline(baseline=(config.T_MIN, 0))
        self.epochsData.save(filepath, overwrite=True)
        
        print(f" Saved epoched data to: {filepath}")
        printSectionFooter("✅ Epoched Data Saving Complete ✅")


class ExtractEpochs:
    def __init__(self,
        subjectId=None, sessionId=None, 
            runId='01', taskName='PilotStudy', 
            bidsDir=config.BIDS_DIR, 
            speechType=config.SPEECH_TYPE, 
            languageElement=config.LANGUAGE_ELEMENT,
            startEnd = config.START_END,
            eventType=config.EVENT_TYPE, 
            trialPhase=config.TRIAL_PHASE, 
            presentationMode=config.PRESENTATION_MODE,
        ):

        printSectionHeader(" Extracting Epochs Data Initialization ")

        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        self.speechType = speechType
        self.languageElement = languageElement
        self.startEnd = startEnd
        self.eventType = eventType
        self.trialPhase = trialPhase
        self.presentationMode = presentationMode 
        self.extactReleventEvents()

    def checkIfEpochsFileExist(self):
        filename, folder, destinationDir = getFolderAndDestination(
            self.subjectId, self.sessionId, self.taskName,
            self.runId, self.speechType, self.languageElement,
            self.eventType, self.startEnd, self.trialPhase,
            self.presentationMode
        )
        self.filename = filename
        self.folder = folder
        self.destinationDir = destinationDir

        self.filepath = Path(self.destinationDir, filename)
        if os.path.exists(self.filepath):
            print(f"{Fore.YELLOW}🔍 Epochs file found.{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.YELLOW}🔍 Epochs file not found. Creating new epochs...{Style.RESET_ALL}")
            return False

    def extactReleventEvents(self):
        if not self.checkIfEpochsFileExist():
            self.neuralData = NeuralDatasetExtractor(
                    subjectId=self.subjectId, sessionId=self.sessionId, runId=self.runId,
                    taskName=self.taskName, bidsDir=self.bidsDir,
                    speechType=self.speechType, languageElement=self.languageElement,
                    eventType=self.eventType, trialPhase=self.trialPhase,
                    presentationMode=self.presentationMode
                )
            self.epochsData = self.neuralData.epochsData
        else:
            print(f"{Fore.GREEN}📂 Loading existing epochs from {self.filepath}{Style.RESET_ALL}")
            self.epochsData = mne.read_epochs(self.filepath, preload=True)

def extractEpochsDataForAllSubjects():
    subjectDirs = [d for d in os.listdir(config.BIDS_DIR) if d.startswith('sub')]

    for subjectDir in subjectDirs:
        subjectId = subjectDir.split('-')[1]
        subjectDirPath = Path(config.BIDS_DIR, subjectDir)
        sessionDirs = [d for d in os.listdir(subjectDirPath) if d.startswith('ses')]
        for session in sessionDirs:
            sessionId = session.split('-')[1]

            try:
                ExtractEpochs(
                    subjectId=subjectId,
                    sessionId=sessionId
                )
            except:
                print(f'Error creating events file for sub-{subjectId}, ses-{sessionId}')


class RealSilentData:
    def __init__(self, subjectId, sessionId) -> None:
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.extractRealSilentEvents()

    
    def extractRealSilentEvents(self):
       
        result, filename, folder, destination = checkIfEpochsFileExist(
            self.subjectId, self.sessionId
        )
        self.filename = filename
        self.folder = folder
        self.destinationDir = destination
        if not result:
            print('Error no Epochs File Found')
        else:
            self.filepath = Path(self.destinationDir, self.filename)
            print(f"{Fore.GREEN}📂 Loading existing epochs from {self.filepath}{Style.RESET_ALL}")
            self.epochsData = mne.read_epochs(self.filepath, preload=True)
           
            events = self.epochsData.events
            eventIds = self.epochsData.event_id
            eventIdsReversed = {str(value): key for key, value in eventIds.items()}

            silentIndexes, realIndexes = [], []
            for index in range(len(events)):
                if 'Real' in eventIdsReversed[str(events[index][2])]:
                    silentIndexes.append(index)
                if 'Silent' in eventIdsReversed[str(events[index][2])]:
                    realIndexes.append(index)
            
            self.realData = self.epochsData[realIndexes]
            self.silentData = self.epochsData[silentIndexes]
        realData = self.realData.get_data(copy=True)
        silentData = self.silentData.get_data(copy=True)

        self.data = np.concatenate((realData, silentData), axis=0)
        self.labels = np.array([0]* realData.shape[0] + [1]*silentData.shape[0])


class VowelDataExtractor:
    """
    A class for extracting and processing Vowel data from neural datasets.

    This class provides methods to extract and process data for different groups
    based on specified categories. It utilizes the NeuralDatasetExtractor class
    to handle the initial data extraction and preprocessing.

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
        groupCategories (list): A list of group categories to be used for data extraction.
        neuralData (NeuralDatasetExtractor): An instance of the NeuralDatasetExtractor class.
        groupedData (dict): A dictionary to store the extracted data for each group.
        groupCategories (list): A list of group categories to be used for data extraction.
        neuralData (NeuralDatasetExtractor): An instance of the NeuralDatasetExtractor class.
        groupedData (dict): A dictionary to store the extracted data for each group.
    """

    def __init__(self, 
        subjectId=None, sessionId=None, 
            runId='01', taskName='PilotStudy', 
            bidsDir=config.BIDS_DIR, 
            speechType=config.SPEECH_TYPE, 
            languageElement=config.LANGUAGE_ELEMENT,
            startEnd = config.START_END,
            eventType=config.EVENT_TYPE, 
            trialPhase=config.TRIAL_PHASE, 
            presentationMode=config.PRESENTATION_MODE,
            groupCategories=['a', 'e', 'i', 'o', 'u']
        ):
        printSectionHeader(" Initializing VowelDataExtractor ")
        
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.bidsDir = bidsDir
        self.speechType = speechType
        self.languageElement = languageElement
        self.startEnd = startEnd
        self.eventType = eventType
        self.trialPhase = trialPhase
        self.presentationMode = presentationMode 
        self.groupCategories = groupCategories
        self.frequencyRange = np.logspace(*np.log10([1, 120]), num=10)
        self.morletFeatures = None
        self.timings = {}  
        self.loadEpochsData()
        self.displayGroupInfo()
        printSectionFooter("✅ VowelDataExtractor Initialization Complete ✅")
        self.extractVowelData()
        pdb.set_trace()
    
    def loadEpochsData(self):
        """
        Load epoched data from a file or create new epochs if the file doesn't exist.

        This method attempts to load pre-existing epoched data from a file. If the file
        doesn't exist, it creates a new NeuralDatasetExtractor instance to generate the epochs file.

        Returns:
            None
        """
        start_time = time.time()
        epochsExtractor = ExtractEpochs(
            subjectId=self.subjectId, sessionId=self.sessionId,
        )
        
        self.epochsData = epochsExtractor.epochsData

        self.eventIds = self.epochsData.event_id
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}⏱️ Time taken to load epochs: {timedelta(seconds=duration)}{Style.RESET_ALL}")

    def extractVowelData(self):
        start_time = time.time()
        printSectionHeader(f"{Fore.MAGENTA}🔬 Extracting Vowels Data 🔬{Style.RESET_ALL}")
        vowelIndexDict = {vowel: [] for vowel in self.groupCategories}
        self.groupedData = {vowel: [] for vowel in self.groupCategories}

        for index, (_, _, event_id) in enumerate(self.epochsData.events):
            eventName = self.eventIdsReversed[event_id]
            for vowel in self.groupCategories:
                if eventName.lower().endswith(vowel.lower()):
                    vowelIndexDict[vowel].append(index)
                    break

        for vowel in self.groupCategories:
            self.groupedData[vowel] = self.epochsData[vowelIndexDict[vowel]].get_data(copy=True)
            print(f"{Fore.CYAN}📊 Vowel {vowel}: shape {self.groupedData[vowel].shape}{Style.RESET_ALL}")

    def displayGroupInfo(self):
        """
        Display information about the group categories and other relevant details.

        This method prints out the subject ID, session ID, run ID, data folder,
        and the group categories in a visually appealing format.

        Returns:
            None
        """
        
        printSectionHeader("ℹ️ Subject, Session, Task, and Run Information ℹ️")
        
        color = Fore.MAGENTA
        size= '\033[4m'
        
        print(f"{color}{size}🧑‍🔬 Subject ID: {self.subjectId}".ljust(60))  
        print(f"{color}\033[1m📅 Session ID: {self.sessionId}".ljust(60))   
        print(f"{color}\033[1m🏃‍♂️ Run ID: {self.runId}".ljust(60))         
        print(f"{color}\033[1m📝 Task Name: {self.taskName}".ljust(60))    
        print(f"{color}\033[1m🔊 Speech Type: {self.speechType}".ljust(60)) 
        print(f"{color}\033[1m🔤 Language Element: {self.languageElement}".ljust(60)) 
        print(f"{color}\033[1m📊 Event Type: {self.eventType}".ljust(60)) 
        print(f"{color}\033[1m⏳ Start/End: {self.startEnd}".ljust(60))   
        print(f"{color}\033[1m⏳ Trial Phase: {self.trialPhase}".ljust(60))   
        print(f"{color}\033[1m🖥️ Presentation Mode: {self.presentationMode}".ljust(60)) 
         
       
        printSectionFooter("✅ Group Information Display Complete ✅")

   
        """
        Execute the complete data processing pipeline.

        This method calls the necessary functions to compute Morlet features,
        extract data for groups, and create the training dataset.

        Returns:
            None
        """
        overall_start_time = time.time()
        printSectionHeader(f"{Fore.GREEN}🔄 Starting Data Processing Pipeline 🔄{Style.RESET_ALL}")
        self.computeMorletFeatures()
        self.categorizeDataByVowels()
        self.makeTrainDataset()
        overall_duration = time.time() - overall_start_time
        print(f"\n{Fore.CYAN}⏱️ Total processing time: {timedelta(seconds=overall_duration)}{Style.RESET_ALL}")
        printSectionFooter(f"{Fore.GREEN}✅ Data Processing Complete ✅{Style.RESET_ALL}")
