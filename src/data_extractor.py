import src.config as config
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
from src.utils import printSectionFooter, printSectionHeader

class NeuralDatasetExtractor:
    """
    A class for processing neural data from EEG experiments.

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
        self.frequencyRange = np.logspace(*np.log10([1, 120]), num=10)

        self.bidsFilepath = BIDSPath(
            subject=self.subjectId, session=self.sessionId, run=self.runId,
            task=self.taskName, suffix='eeg', root=self.bidsDir
        )

        printSectionHeader("📊 Loading Data 📊")
        self.rawData = read_raw_bids(self.bidsFilepath)
        self.channels = self.rawData.ch_names
        self.rawData.load_data()
        printSectionFooter("✅ Data Loading Complete ✅")
    
        self.events, self.eventIds = mne.events_from_annotations(self.rawData, verbose=False)
        self.eventIdsReversed = {value: key for key, value in self.eventIds.items()}
        
        printSectionFooter("🎉 Initialization Complete 🎉")
        self.getRealtedEvents()
        self.extractEvents()
        
    def preprocessData(self):
        """
        Apply preprocessing steps to the raw EEG data.

        This method applies the following preprocessing steps:
        1. Notch filter at 50 Hz and 100 Hz to remove power line noise.
        2. Bandpass filter between 0.1 Hz and 120 Hz.
        3. Independent Component Analysis (ICA) to capture 99% of the variance.

        Returns:
            None
        """
        printSectionHeader("🧹 Applying Preprocessing Steps 🧹")
        
        print("1️⃣ Applying notch filter...")
        self.rawData.notch_filter([50, 100])
        
        print("2️⃣ Applying bandpass filter...")
        self.rawData.filter(l_freq=0.1, h_freq=120)

        print("3️⃣ Performing ICA...")
        ica = mne.preprocessing.ICA(max_iter='auto', random_state=42)
        rawDataForICA = self.rawData.copy()
        ica.fit(rawDataForICA)
        ica.apply(self.rawData)
        
        print("4️⃣ Setting EEG reference...")
        self.rawData.set_eeg_reference(ref_channels=['FCz'])

        printSectionFooter("✨ Preprocessing Completed ✨")

    def getRealtedEvents(self):
        """
        Extract events of interest based on the specified parameters.

        This method filters the events based on the speech type, language element,
        event type, trial phase, and presentation mode specified during initialization.

        Returns:
            None
        """
        printSectionHeader("🔍 Extracting Related Events 🔍")
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
        print(f"📊 Found {len(self.intrestedEvents)} events of interest")
        printSectionFooter("✅ Event Extraction Complete ✅")
    
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
        print(f"📊 Created {len(self.epochsData)} epochs")
        printSectionFooter("✅ Epoch Creation Complete ✅")

    def computeMorletFeatures(self):
        """
        Compute time-frequency representations using Morlet wavelets.

        This method computes the time-frequency representations of the epochs
        using Morlet wavelets.

        Returns:
            None
        """
        printSectionHeader("🌊 Computing Morlet Features 🌊")
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
    
