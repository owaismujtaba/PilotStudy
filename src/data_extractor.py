import src.config as config
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
from src.utils import printSectionFooter, printSectionHeader
from pathlib import Path
import os
from colorama import Fore, Style, init
import pdb

init(autoreset=True)

class NeuralDatasetExtractor:
    """
    A class for processing neural data from EEG experiments.
    A class for processing neural data from EEG experiments.

    This class provides methods to load, process, and extract EEG data
    from BIDS-formatted datasets. It includes functionality for initializing
    the dataset, extracting event information, and creating epochs for words
    and syllables based on various parameters such as speech type, language element,
    This class provides methods to load, process, and extract EEG data
    from BIDS-formatted datasets. It includes functionality for initializing
    the dataset, extracting event information, and creating epochs for words
    and syllables based on various parameters such as speech type, language element,
    event type, trial phase, and presentation mode.

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
        intrestedEvents (numpy.ndarray): The filtered events based on specified parameters.
        epochsData (mne.Epochs): Epochs object for the filtered events.
        morletFeatures (mne.time_frequency.EpochsTFR): Time-frequency representations of the epochs.
        intrestedEvents (numpy.ndarray): The filtered events based on specified parameters.
        epochsData (mne.Epochs): Epochs object for the filtered events.
        morletFeatures (mne.time_frequency.EpochsTFR): Time-frequency representations of the epochs.
    """

    def __init__(self, 
            subjectId='01', sessionId='01', runId='01', 
            taskName='PilotStudy', bidsDir=config.bidsDir, 
            speechType=None, languageElement=None,
            eventType=None, trialPhase=None, presentationMode=None
        ):
        
        printSectionHeader(f"{Fore.CYAN}🚀 Initializing NeuralDatasetExtractor 🚀{Style.RESET_ALL}")
        
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

        printSectionHeader(f"{Fore.YELLOW}📊 Loading Raw Data 📊{Style.RESET_ALL}")
        self.rawData = read_raw_bids(self.bidsFilepath)
        self.channels = self.rawData.ch_names
        self.rawData.load_data()
        printSectionFooter(f"{Fore.GREEN}✅ Data Loading Complete ✅{Style.RESET_ALL}")
        self.preprocessData()
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        
        printSectionFooter(f"{Fore.GREEN}🎉 NeuralDatasetExtractor Initialization Complete 🎉{Style.RESET_ALL}")
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
        print(f"🧑‍🔬 Subject ID: {self.subjectId}".center(60))
        print(f"📅 Session ID: {self.sessionId}".center(60))
        print(f"🏃‍♂️ Run ID: {self.runId}".center(60))
        print(f"📝 Task Name: {self.taskName}".center(60))
    
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
        printSectionHeader(f"{Fore.MAGENTA}🧹 Applying Preprocessing Steps 🧹{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}1️⃣ Applying notch filter...{Style.RESET_ALL}")
        self.rawData.notch_filter([50, 100])
        
        print(f"{Fore.YELLOW}2️⃣ Applying bandpass filter...{Style.RESET_ALL}")
        self.rawData.filter(l_freq=0.1, h_freq=120, n_jobs=config.nJobs)

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
        printSectionHeader(f"🔍 Extracting Related Events 🔍")
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
        print(f"{Fore.MAGENTA}📊 Found {len(self.intrestedEvents)} events of interest{Style.RESET_ALL}".center(60))
        printSectionFooter(f"✅ Event Extraction Complete ✅")
    
    def extractEvents(self):
        """
        Create epochs from the filtered events.

        This method creates epochs from the events of interest using the specified
        time window and baseline correction.

        Returns:
            None
        """
        printSectionHeader("🎭 Creating Epochs 🎭")
        self.epochsData = mne.Epochs(
            self.rawData, 
            events=self.intrestedEvents, 
            tmin=config.tmin, tmax=config.tmax,
            baseline=(-0.5, 0), 
            picks=self.channels, 
            preload=True,
            verbose=False
        )

        eventIds = self.epochsData.event_id
        
        newEventIds = {}
        
        for key, item in eventIds.items():
            newEventIds[key] = self.eventIdsReversed[int(key)]
        self.epochsData.event_id = newEventIds

        print(f"{Fore.GREEN}📊 Created {len(self.epochsData)} epochs{Style.RESET_ALL}".center(60))
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
        
        printSectionHeader("💾 Saving Epoched Data 💾")

        dataDir = config.dataDir
        folder = f'{self.speechType}{self.languageElement}{self.eventType}{self.trialPhase}{self.presentationMode}'
        filename = f"sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}_epo.fif"
        destinationDir = Path(dataDir, f'sub-{self.subjectId}', f'ses-{self.sessionId}', folder)

        Path(destinationDir).mkdir(parents=True, exist_ok=True)
        
        filepath = Path(destinationDir) / filename
        self.epochsData.save(filepath, overwrite=True)
        
        print(f"📁 Saved epoched data to: {filepath}")
        printSectionFooter("✅ Epoched Data Saving Complete ✅")

class GroupDataExtractor:
    """
    A class for extracting and processing group data from neural datasets.

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
        subjectId='01', sessionId='01', runId='01', 
        taskName='PilotStudy', bidsDir=config.bidsDir, 
        speechType=None, languageElement=None,
        eventType=None, trialPhase=None, presentationMode=None,
        groupCategories=['a', 'e', 'i', 'o', 'u']
    ):
        printSectionHeader("🚀 Initializing GroupDataExtractor 🚀")
        
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
        
                
        dataDir = config.dataDir
        folder = f'{self.speechType}{self.languageElement}{self.eventType}{self.trialPhase}{self.presentationMode}'
        filename = f"sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}_epo.fif"
        destinationDir = Path(dataDir, f'sub-{self.subjectId}', f'ses-{self.sessionId}', folder)
        filepath = Path(destinationDir, filename)

        if not os.path.exists(filepath):
            self.neuralData = NeuralDatasetExtractor(
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
        else:
            printSectionHeader(f"Loading epochs data from ")
            printSectionHeader(f"File: {filepath}")
            self.loadEpochsDataFromFile(filepath)
        
        printSectionFooter("✅ GroupDataExtractor Initialization Complete ✅")
        
        self.displayGroupInfo()
        self.computeMorletFeatures()
        self.extractDataForGroups()
        #self.makeTrainDataset()
    def loadEpochsDataFromFile(self, filepath):
        self.neuralData = mne.read_epochs(filepath, preload=True)
       
    def extractDataForGroups(self):
        """
        Extract data for each group based on the specified group categories.

        This method extracts the data for each group by filtering the events
        based on the group categories and stores the extracted data in a dictionary.

        Returns:
            None
        """
        printSectionHeader("🔍 Extracting Data for Groups 🔍")
        groupsEventIndexs = {group: [] for group in self.groupCategories}
        groupsEvenData = {group: [] for group in self.groupCategories}

        for index in range(len(self.neuralData.morletFeatures.events)):
            eventId = self.neuralData.morletFeatures.events[index][2]
            eventName = self.neuralData.eventIdsReversed[eventId]
            for key in groupsEventIndexs:
                if eventName.endswith(key) or eventName.endswith(key.upper()):
                    groupsEventIndexs[key].append(index)
                    break
                
        for group in self.groupCategories:
            groupsEvenData[group] = self.neuralData.morletFeatures[groupsEventIndexs[group]].get_data()

        self.groupedData = groupsEvenData
        print(f"{Fore.MAGENTA}📊 Extracted data for groups: {', '.join(self.groupCategories)}{Style.RESET_ALL}")
        printSectionFooter("✅ Group Data Extraction Complete ✅")

    def displayGroupInfo(self):
        """
        Display information about the group categories.

        This method prints out the details of the group categories in a clear
        and visually appealing format.

        Returns:
            None
        """
        printSectionHeader("ℹ️ Group Categories Information ℹ️")
        print(f"{Fore.MAGENTA}📊 Group Categories: {', '.join(self.groupCategories)}{Style.RESET_ALL}")
        printSectionFooter("✅ Group Information Display Complete ✅")
    
    def computeMorletFeatures(self):
        """
        Compute time-frequency representations using Morlet wavelets.

        This method computes the time-frequency representations of the epochs
        using Morlet wavelets.

        Returns:
            None
        """
        printSectionHeader("🌊 Computing Morlet Features 🌊")
        pdb.set_trace()
        self.morletFeatures = self.epochsData.compute_tfr(
            method='morlet',
            freqs=self.frequencyRange,
            n_cycles=self.frequencyRange/8,
            use_fft=True,
            return_itc=False,
            average=False
        )
        print(f"📊 Computed Morlet features for {len(self.morletFeatures)} epochs")
        printSectionFooter("✅ Morlet Feature Computation Complete ✅")
    
    def makeTrainDataset(self):
        data = None
        labels = None
        for group in self.groupCategories:
            if data == None:
                data = self.groupedData[group]
                labels = [group]*data.shape[0]
            else:
                data = np.concatenate((data, self.groupedData[group]), axis=0)
                labels += [group]*self.groupedData[group].shape[0]


        self.xTrain = data
        self.yTrain = labels
        
