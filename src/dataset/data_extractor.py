import os
import mne
import time
from datetime import timedelta
import numpy as np
from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids
from colorama import Fore, Style, init

import src.utils.config as config
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
        self.preprocessData()
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        
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
        printSectionHeader("ℹ️ Subject, Session, Task, and Run Information ℹ️")
        print(f"{Fore.CYAN}🧑‍🔬 Subject ID: {self.subjectId}".center(60))  
        print(f"📅 Session ID: {self.sessionId}".center(60))   
        print(f"🏃‍♂️ Run ID: {self.runId}".center(60))         
        print(f"📝 Task Name: {self.taskName}".center(60))    
        print(f"🔊 Speech Type: {self.speechType}".center(60)) 
        print(f"🔤 Language Element: {self.languageElement}".center(60)) 
        print(f"📊 Event Type: {self.eventType}".center(60))   
        print(f"⏳ Trial Phase: {self.trialPhase}".center(60))   
        print(f"🖥️ Presentation Mode: {self.presentationMode}".center(60)) 
        print(f"📂 BIDS Directory: {self.bidsDir}{Style.RESET_ALL}")  
    
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

        print(f"{Fore.YELLOW}3️⃣ Performing ICA...{Style.RESET_ALL}")
        ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
        rawDataForICA = self.rawData.copy()
        ica.fit(rawDataForICA)
        ica.apply(self.rawData)
        
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
            if (self._checkSpeechType(eventName, self.speechType) and 
                self._checkLanguageElement(eventName, self.languageElement) and 
                self._checkEventType(eventName, self.eventType) and 
                self._checkTrialPhase(eventName, self.trialPhase) and 
                self._checkPresentationMode(eventName, self.presentationMode)
            ):
                intrestedEvents.append(self.events[index])
        self.intrestedEvents = np.array(intrestedEvents)
        print(f"{Fore.MAGENTA} Found {len(self.intrestedEvents)} events of interest{Style.RESET_ALL}".center(60))
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

    
    def _checkSpeechType(self, eventName, speechType):
        """
        Check if the event matches the specified speech type.

        Args:
            eventName (str): The name of the event to check.
            speechType (str): The type of speech to check against.

        Returns:
            bool: True if the event matches the speech type, False otherwise.
        """
        if speechType is None:
            return True
        return speechType in eventName

    def _checkLanguageElement(self, eventName, languageElement):
        """
        Check if the event matches the specified language element.

        Args:
            eventName (str): The name of the event to check.
            languageElement (str): The language element to check against.

        Returns:
            bool: True if the event matches the language element, False otherwise.
        """
        if languageElement is None:
            return True
        return languageElement in eventName

    def _checkEventType(self, eventName, eventType):
        """
        Check if the event matches the specified event type.

        Args:
            eventName (str): The name of the event to check.
            eventType (str): The type of event to check against.

        Returns:
            bool: True if the event matches the event type, False otherwise.
        """
        if eventType is None:
            return True
        return eventType in eventName

    def _checkTrialPhase(self, eventName, trialPhase):
        """
        Check if the event matches the specified trial phase.

        Args:
            eventName (str): The name of the event to check.
            trialPhase (str): The trial phase to check against.

        Returns:
            bool: True if the event matches the trial phase, False otherwise.
        """
        if trialPhase is None:
            return True
        return trialPhase in eventName
    
    def _checkPresentationMode(self, eventName, presentationMode):
        """
        Check if the event matches the specified presentation mode.

        Args:
            eventName (str): The name of the event to check.
            presentationMode (str): The presentation mode to check against.

        Returns:
            bool: True if the event matches the presentation mode, False otherwise.
        """
        if presentationMode is None:
            return True
        return presentationMode in eventName
    
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

        dataDir = config.BIDS_DIR
        folder = f'{self.speechType}{self.languageElement}{self.eventType}{self.trialPhase}{self.presentationMode}'
        filename = f"sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}_epo.fif"
        destinationDir = Path(dataDir, f'sub-{self.subjectId}', f'ses-{self.sessionId}', folder)

        Path(destinationDir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(destinationDir) / filename
        #self.epochsData.apply_baseline(baseline=(config.T_MIN, 0))
        self.epochsData.save(filepath, overwrite=True)
        
        print(f" Saved epoched data to: {filepath}")
        printSectionFooter("✅ Epoched Data Saving Complete ✅")





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
            eventType=config.EVENT_TYPE, trialPhase=None, 
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
    
    def loadEpochsData(self):
        """
        Load epoched data from a file or create new epochs if the file doesn't exist.

        This method attempts to load pre-existing epoched data from a file. If the file
        doesn't exist, it creates a new NeuralDatasetExtractor instance to generate the epochs file.

        Returns:
            None
        """
        start_time = time.time()
        dataDir = config.DATA_DIR
        folder = f'{self.speechType}{self.languageElement}{self.eventType}{self.trialPhase}{self.presentationMode}'
        filename = f"sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}_epo.fif"
        destinationDir = Path(dataDir, f'sub-{self.subjectId}', f'ses-{self.sessionId}', folder)
        filepath = Path(destinationDir, filename)
        self.groupFolder = folder
        if not os.path.exists(filepath):
            print(f"{Fore.YELLOW}🔍 Epochs file not found. Creating new epochs...{Style.RESET_ALL}")
            self.neuralData = NeuralDatasetExtractor(
                subjectId=self.subjectId, sessionId=self.sessionId, runId=self.runId,
                taskName=self.taskName, bidsDir=self.bidsDir,
                speechType=self.speechType, languageElement=self.languageElement,
                eventType=self.eventType, trialPhase=self.trialPhase,
                presentationMode=self.presentationMode
            )
            self.epochsData = self.neuralData.epochsData
        else:
            print(f"{Fore.GREEN}📂 Loading existing epochs from {filepath}{Style.RESET_ALL}")
            self.epochsData = mne.read_epochs(filepath, preload=True)

        self.eventIds = self.epochsData.event_id
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}⏱️ Time taken to load epochs: {timedelta(seconds=duration)}{Style.RESET_ALL}")

    def displayGroupInfo(self):
        """
        Display information about the group categories and other relevant details.

        This method prints out the subject ID, session ID, run ID, data folder,
        and the group categories in a visually appealing format.

        Returns:
            None
        """
        printSectionHeader("ℹ️ Group Categories Information ℹ️")
        print(f"{Fore.BLUE}🧑 Subject ID:               {self.subjectId}{Style.RESET_ALL}".ljust(60))
        print(f"{Fore.GREEN}📅 Session ID:               {self.sessionId}{Style.RESET_ALL}".ljust(60))
        print(f"{Fore.YELLOW}🏃‍♂️ Run ID:                   {self.runId}{Style.RESET_ALL}".ljust(60))
        print(f"{Fore.MAGENTA}📁 Data Folder:               {self.groupFolder}{Style.RESET_ALL}".ljust(60))
        print(f"{Fore.CYAN}📊 Group Categories: {', '.join(self.groupCategories)}{Style.RESET_ALL}".ljust(60))
        printSectionFooter("✅ Group Information Display Complete ✅")

    def computeMorletFeatures(self):
        """
        Compute Morlet wavelet features for the epoched data.

        This method calculates time-frequency representations using Morlet wavelets
        for the epoched data across the specified frequency range.

        Returns:
            None
        """
        start_time = time.time()
        printSectionHeader(f"{Fore.CYAN}🌊 Computing Morlet Features 🌊{Style.RESET_ALL}")
        self.morletFeatures = self.epochsData.compute_tfr(
            method='morlet',
            freqs=self.frequencyRange,
            n_cycles=self.frequencyRange/8,
            use_fft=True,
            return_itc=False,
            average=False
        )
        print(f"{Fore.GREEN}✨ Computed Morlet features for {len(self.morletFeatures)} epochs{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📊 Shape: {self.morletFeatures.get_data().shape}{Style.RESET_ALL}")
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}⏱️ Time taken to compute Morlet features: {timedelta(seconds=duration)}{Style.RESET_ALL}")
        printSectionFooter(f"{Fore.GREEN}✅ Morlet Feature Computation Complete ✅{Style.RESET_ALL}")

    def categorizeDataByVowels(self):
        """
        Categorize Morlet and raw features for each vowel group.

        This method separates the computed Morlet features and raw data into groups
        based on the specified vowel categories (a, e, i, o, u).

        Returns:
            None
        """
        start_time = time.time()
        printSectionHeader(f"{Fore.MAGENTA}🔬 Categorizing Features by Vowels 🔬{Style.RESET_ALL}")
        vowelIndexDict = {vowel: [] for vowel in self.groupCategories}
        self.groupedMorletData = {vowel: [] for vowel in self.groupCategories}
        self.groupedRawData = {vowel: [] for vowel in self.groupCategories}

        for index, (_, _, event_id) in enumerate(self.morletFeatures.events):
            eventName = self.eventIdsReversed[event_id]
            for vowel in self.groupCategories:
                if eventName.lower().endswith(vowel.lower()):
                    vowelIndexDict[vowel].append(index)
                    break

        for vowel in self.groupCategories:
            if self.morletFeatures:
                self.groupedMorletData[vowel] = self.morletFeatures[vowelIndexDict[vowel]].get_data()
            self.groupedRawData[vowel] = self.epochsData[vowelIndexDict[vowel]].get_data()
            print(f"{Fore.CYAN}📊 Vowel {vowel}: Morlet shape {self.groupedMorletData[vowel].shape}, Raw shape {self.groupedRawData[vowel].shape}{Style.RESET_ALL}")

        print(f"{Fore.MAGENTA} Categorized data for vowels: {', '.join(self.groupCategories)}{Style.RESET_ALL}")
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}⏱️ Time taken to categorize data by vowels: {timedelta(seconds=duration)}{Style.RESET_ALL}")
        printSectionFooter("✅ Vowel Data Categorization Complete ✅")

    def makeTrainDataset(self, test_size=0.3):
        """
        Create training and testing datasets from the extracted features.

        This method combines the grouped data, adjusts the time window, and splits
        the data into training and testing sets for both Morlet and raw features.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.

        Returns:
            None
        """
        start_time = time.time()
        printSectionHeader(f"{Fore.YELLOW}🚂 Creating Training Dataset 🚂{Style.RESET_ALL}")
        morlet_features, raw_features, labels = [], [], []

        for index, group in enumerate(self.groupCategories):
            morlet_data = self.groupedMorletData[group]
            raw_data = self.groupedRawData[group]
            
            morlet_features.append(morlet_data)
            raw_features.append(raw_data)
            labels.extend([index] * len(morlet_data))
            
            print(f"{Fore.MAGENTA} Group {group}: Morlet shape {morlet_data.shape}, Raw shape {raw_data.shape}{Style.RESET_ALL}")

        self.morletFeatures = np.concatenate(morlet_features)
        self.rawFeatures = np.concatenate(raw_features)
        self.labels = np.array(labels)

        # Adjust time window
        timeStart = int(abs(config.T_MIN * 1000))
        self.morletFeatures = self.morletFeatures[:, :, :, timeStart:]
        self.rawFeatures = self.rawFeatures[:, :, timeStart:]

        # Split into train and test sets
        from sklearn.model_selection import train_test_split
        self.X_train_morlet, self.X_test_morlet, self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            self.morletFeatures, self.rawFeatures, self.labels, test_size=test_size, stratify=self.labels, random_state=42
        )

        print(f"{Fore.GREEN}✅ Dataset created with {len(self.labels)} samples{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Morlet features shape: {self.morletFeatures.shape}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Raw features shape: {self.rawFeatures.shape}{Style.RESET_ALL}")
        duration = time.time() - start_time
        print(f"{Fore.YELLOW}⏱️ Time taken to create training dataset: {timedelta(seconds=duration)}{Style.RESET_ALL}")
        printSectionFooter("✅ Dataset Creation Complete ✅")

    def process_data(self):
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
