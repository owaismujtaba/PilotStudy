from pathlib import Path
import os
import mne
import csv
import json
from mne_bids import BIDSPath, write_raw_bids
import numpy as np
import pyxdf
from datetime import datetime
from collections import Counter
import src.config as confg
from scipy.io.wavfile import write

import src.config as config
import pdb

class XDFData:
    """
    This class handles data from an XDF file (e.g., EEG and audio).
    It processes the data, sets up EEG and audio metadata, and allows exporting
    to BIDS formats.
    """
    def __init__(self, 
            filepath=None, 
            subjectId='01', sessionId='01', 
            runId='01', taskName='PilotStudy'
        ) -> None:
        """
        Initialize the XDFData object with the provided file path and identifiers.

        Parameters:
        - filepath: Path to the XDF file.
        - subjectId: Subject identifier (default: '01').
        - sessionId: Session identifier (default: '01').
        - runId: Run identifier (default: '01').
        - taskName: Task name (default: 'PilotStudy').

        Returns: None
        """
        self.filepath = filepath
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.rawData = None
        self.destinationDir = Path(f'{config.bidsDir}/sub-{self.subjectId}/ses-{self.sessionId}')
        self.fileName = f'sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}'
        
        self.bidsPath = BIDSPath(
            subject=self.subjectId, session=self.sessionId,
            task=self.taskName, run=self.runId, datatype='eeg', root=config.bidsDir
        ) 
        
        self.loadXdfData()
        self.setupData()
        self.printInfo()
        
        self.createMneRawObjectForEeg()
        self.makeAnnotations()
        
        self.createEDFFile()
        self.createAudio()
        self.createEventsFileForAudio()
        self.addNormalizationInfoToSidecar()
        
    def loadXdfData(self):
        """
        Load XDF data from the file.
        """
        print('Loading XDF Data')
        self.data, self.header = pyxdf.load_xdf(self.filepath)

    def findDataStreamIndexs(self):
        """
        Find and set the indices for different data streams in the XDF file.
        """
        for index, stream in enumerate(self.data):
            streamType = stream['info']['type'][0]
            streamName = stream['info']['name'][0]

            if streamName == 'SingleWordsMarkerStream':
                self.markersIndex = index
            elif streamType == 'EEG':
                self.eegDataIndex = index
            elif streamType == 'Audio':
                self.audioDataIndex = index

    def setupData(self):
        """
        Extract EEG, markers, and audio data from the XDF file.
        Set up channel names and timestamps.
        """
        print('Setting Up XDF Data')
        self.eegChannelNames = []
        self.measDate = self.header['info']['datetime'][0]
        self.findDataStreamIndexs()
        self.markers = self.data[self.markersIndex]['time_series']
        self.markersTimestamps = self.data[self.markersIndex]['time_stamps']
        self.markers.pop()
        self.markers = [marker[0] for marker in self.markers]
        self.eegData = self.data[self.eegDataIndex]['time_series']
        self.eegSamplingFrequency = int(float(self.data[self.eegDataIndex]['info']['nominal_srate'][0]))
        self.eegTimestamps = self.data[self.eegDataIndex]['time_stamps']
        self.audioData = self.data[self.audioDataIndex]['time_series']
        self.audioTimestamps = self.data[self.audioDataIndex]['time_stamps']
        self.audioSamplingFrequency = int(float(self.data[3]['info']['nominal_srate'][0]))
        channelNames = self.data[self.eegDataIndex]['info']['desc'][0]['channels'][0]['channel']
        for item in channelNames:
            self.eegChannelNames.append(item['label'][0]) 
    
    def printInfo(self):
        """
        Print information about the loaded data.
        """
        print(f'No of Markers: {len(self.markers)} No .of Marker Timestamps: {self.markersTimestamps.shape[0]}')
        print(f'EEG data Shape: {self.eegData.shape} No .of eeg Timestamps: {self.eegTimestamps.shape[0]}')
        print(f'Audio data Shape: {self.audioData.shape} No .of audio Timestamps: {self.audioTimestamps.shape[0]}')
        print('EEG Channels:', self.eegChannelNames)
        print(f'Sampling Frequency ::: EEG: {self.eegSamplingFrequency}, Audio: {self.audioSamplingFrequency}')
    
    def createMneRawObjectForEeg(self):
        """
        Create an MNE Raw object for EEG data to facilitate further processing.
        The EEG data is normalized and converted to MNE format.
        """
        print('Creating MNE Data')
        meas_date = datetime.strptime(self.measDate, '%Y-%m-%dT%H:%M:%S%z')
        meas_date = (int(meas_date.timestamp()), int(meas_date.microsecond))
        info = mne.create_info(
            ch_names=self.eegChannelNames, 
            sfreq=self.eegSamplingFrequency, 
            ch_types='eeg',
        )
        info.set_meas_date(meas_date)
        self.normalizeDataToInt16()
        self.rawEegMNEData = mne.io.RawArray(self.eegData.T, info)

    def normalizeDataToInt16(self):
        """
        Normalize EEG data to a range between 0 and 1, for saving in EDF.
        """
        self.minimum = self.eegData.min()
        self.maximum = self.eegData.max()
        self.eegData = (self.eegData - self.minimum)/(self.maximum - self.minimum)

    def getOnsetCodesForAnnotations(self):
        """
        Extract event codes from markers for creating annotations in the data.

        Returns:
        - onset (list[float]): List of onset times for annotations.
        - codes (list[str]): List of event codes.
        - duration (list[float]): List of event durations.
        """
        print('Setting up Annotation Codes for Data')
        codes = []
        onset = []
        description = []
        duration = []
        markers = self.markers
        markerTimestamps = self.markersTimestamps - self.eegTimestamps[0]
        for index in range(len(markers) - 1):
            marker = markers[index]
            code = self.buildCodeFromMarker(marker)
            codes.append(code)
            onset.append(markerTimestamps[index])
            description.append(marker)
            duration.append(markerTimestamps[index + 1] - markerTimestamps[index])

        return onset, codes, duration

    def buildCodeFromMarker(self, marker):
        """
        Build a code from a marker.

        Parameters:
        - marker (str): The marker to build the code from.

        Returns:
        - str: The built code.
        """
        codeComponents = [
            self.getSpeechType(marker),
            self.getWordType(marker),
            self.getExperimentPhase(marker),
            self.getEventType(marker),
            self.getTrialPhase(marker),
            self.getStimulusType(marker)
        ]
        code = ','.join(codeComponents)
        wordOrSyllable = marker.split(':')[1].split('_')[1]
        return f"{code},{wordOrSyllable}"

    def getSpeechType(self, marker):
        """
        Get the speech type from a marker.

        Parameters:
        - marker (str): The marker to extract the speech type from.

        Returns:
        - str: The speech type.
        """
        if 'Silent' in marker:
            return 'Silent'
        elif 'Real' in marker:
            return 'Overt'
        return ''

    def getWordType(self, marker):
        """
        Get the word type from a marker.

        Parameters:
        - marker (str): The marker to extract the word type from.

        Returns:
        - str: The word type.
        """
        if 'Word' in marker:
            return 'Word'
        elif 'Syllable' in marker:
            return 'Syllable'
        return ''

    def getExperimentPhase(self, marker):
        """
        Get the experiment phase from a marker.

        Parameters:
        - marker (str): The marker to extract the experiment phase from.

        Returns:
        - str: The experiment phase.
        """
        if 'Practice' in marker:
            return 'Practice'
        elif 'Experiment' in marker:
            return 'Experiment'
        return ''

    def getEventType(self, marker):
        """
        Get the event type from a marker.

        Parameters:
        - marker (str): The marker to extract the event type from.

        Returns:
        - str: The event type.
        """
        if 'Start' in marker:
            return 'Start'
        elif 'End' in marker:
            return 'End'
        return ''

    def getTrialPhase(self, marker):
        """
        Get the trial phase from a marker.

        Parameters:
        - marker (str): The marker to extract the trial phase from.

        Returns:
        - str: The trial phase.
        """
        for phase in ['Fixation', 'Stimulus', 'ISI', 'ITI', 'Speech']:
            if phase in marker:
                return phase
        return ''

    def getStimulusType(self, marker):
        """
        Get the stimulus type from a marker.

        Parameters:
        - marker (str): The marker to extract the stimulus type from.

        Returns:
        - str: The stimulus type.
        """
        if 'Audio' in marker:
            return 'Audio'
        elif 'Text' in marker:
            return 'Text'
        elif 'Pictures' in marker:
            return 'Picture'
        return ''
    
    def makeAnnotations(self):
        """
        Create MNE annotations for the EEG data based on the extracted event codes and onsets.
        """
        print('Annotating Data')
        onset, codes, duration = self.getOnsetCodesForAnnotations()
        rawMNEWithAnnotations = self.rawEegMNEData.copy()
        rawMNEWithAnnotations.set_annotations(mne.Annotations(onset=onset, description=codes, duration=duration))

        self.rawMNEWithAnnotations = rawMNEWithAnnotations
        self.onset = onset
        self.codes = codes
        self.duration = duration

    def createAudio(self):
        """
        Create an audio file from the audio data stored in the XDF file.

        Returns:
        - Path: The path of the created audio file.
        """
        print('***************************Creating Audio file***************************')
        destinationDir = self.destinationDir / 'audio'
        self.ensureDirectoryExists(destinationDir)
        audioData = self.audioData
        audioData = audioData.flatten()
        destinationPath = destinationDir / f'{self.fileName}_audio.wav'
        write(str(destinationPath), self.audioSamplingFrequency, audioData)
        print('***************************Audio file created***************************')

        return destinationPath

    def ensureDirectoryExists(self, path):
        """
        Ensure that the given directory exists.

        Parameters:
        path (Path): The path to check/create.
        """
        os.makedirs(path, exist_ok=True)

    def createEventsFileForAudio(self):
        """
        Write synchronized events to a TSV file. This file contains information about 
        events such as onset times, durations, and event types related to the audio data.

        Returns:
        - Path: The path of the created events file.
        """
        print('***************************Writing events to file***************************')
        fileName = f'{self.fileName}_events.tsv'
        bidsHeaders = ['onset', 'duration', 'trial_type']
        destinationDir = self.destinationDir / 'audio'
        self.ensureDirectoryExists(destinationDir)
        fileNameWithPath = destinationDir / fileName       

        with open(fileNameWithPath, "w", newline="") as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=bidsHeaders, delimiter='\t')
            writer.writeheader()

            for index in range(len(self.onset)):
                event = {
                    "onset": self.onset[index],  
                    "duration": self.duration[index],
                    "trial_type": self.codes[index],
                }
                writer.writerow(event)

        print('***************************Events written to file***************************')
        return fileNameWithPath

    def createEDFFile(self):
        """
        Create a BIDS FIF file from the EEG data using the MNE library. The EEG data is
        saved in EDF format as per BIDS specification.
        """
        data = self.rawMNEWithAnnotations
        
        write_raw_bids(data, bids_path=self.bidsPath, allow_preload=True, format='EDF', overwrite=True)
       
        print('***************************BIDS FIF file created***************************')

    def addNormalizationInfoToSidecar(self):
        """
        Add normalization information (minimum and maximum values) to the JSON sidecar file 
        associated with the EEG data. This provides additional metadata about data normalization.
        """
        print('Adding Normalization Info to the Json Side Car')
        filename = f'{self.fileName}_eeg.json'
        fodler = Path(self.destinationDir, 'eeg')
        filepath = Path(fodler, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
        file.close()
        data['Normalization(min, max)'] = f'{self.minimum}, {self.maximum}'

        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
