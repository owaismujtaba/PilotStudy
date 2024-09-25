from pathlib import Path
import os
import csv
from mne_bids import BIDSPath, write_raw_bids
from pyxdf import resolve_streams, match_streaminfos
from mnelab.io.xdf import read_raw_xdf
import src.config as confg
from scipy.io.wavfile import write
import time
import numpy as np
import src.config as config
import pdb
from src.utils import printSectionHeader, printSectionFooter
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()

class XDFData:
    """
    Handles data from an XDF file containing EEG and audio streams.
    Processes the data, sets up metadata, and exports to BIDS-compatible formats.
    
    This class provides functionality to:
    1. Load XDF data containing EEG and audio streams
    2. Process and resample the loaded data
    3. Create BIDS-compatible EDF files for EEG data
    4. Extract and save audio data as WAV files
    5. Generate event files for synchronization
    """

    def __init__(self, 
            filePath=None, 
            subjectId='01', sessionId='01', 
            runId='01', taskName='PilotStudy'
        ) -> None:
        """
        Initializes the XDFData object with file path and identifiers.
        Sets up paths and calls methods to process the data.

        Parameters:
        - filePath: Path to the XDF file containing EEG and audio data
        - subjectId: Unique identifier for the subject (default: '01')
        - sessionId: Identifier for the recording session (default: '01')
        - runId: Identifier for the specific run within a session (default: '01')
        - taskName: Name of the task performed during recording (default: 'PilotStudy')

        """
        self.filePath = filePath
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.rawData = None
        self.eegSamplingFrequency = 1000
        self.audioSamplingFrequency = 48000
        self.destinationDir = Path(f'{config.bidsDir}/sub-{self.subjectId}/ses-{self.sessionId}')
        self.fileName = f'sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}'
        
        self.bidsPath = BIDSPath(
            subject=self.subjectId, session=self.sessionId,
            task=self.taskName, run=self.runId, datatype='eeg', root=config.bidsDir
        ) 
        
        self.loadXdfData()
        self.setupData()
        self.printInfo()
        
        self.createEDFFile()
        self.createAudio()
        self.createEventsFileForAudio()
        
    def loadXdfData(self):
        """
        Loads XDF data from the file, separating EEG and Audio streams.
        Measures and reports the time taken for loading each stream.

        This method performs the following steps:
        1. Resolves streams in the XDF file
        2. Identifies EEG and Audio streams
        3. Loads EEG and Audio data using read_raw_xdf
        """
        printSectionHeader(f"{Fore.CYAN}📂 Loading XDF Data")
        
        startTime = time.time()
        
        streams = resolve_streams(self.filePath)
        print(f"{Fore.YELLOW}🔍 Found {len(streams)} streams in the XDF file")
        eegStreamId = match_streaminfos(streams, [{'type':'EEG'}])[0]
        audioStreamId = match_streaminfos(streams, [{'type':'Audio'}])[0]
        
        print(' Loading EEG stream...'.ljust(30), end='')
        eegStart = time.time()
        self.eegData = read_raw_xdf(self.filePath, stream_ids=[eegStreamId])
        eegTime = time.time() - eegStart
        print(f'✅ Done in {eegTime:.2f} seconds')
        
        print(' Loading Audio stream...'.ljust(30), end='')
        audioStart = time.time()
        self.audioData = read_raw_xdf(self.filePath, stream_ids=[audioStreamId])
        audioTime = time.time() - audioStart
        print(f'✅ Done in {audioTime:.2f} seconds')
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ XDF Data Loading Complete in {totalTime:.2f} seconds")

    def setupData(self):
        """
        Processes loaded EEG and audio data.
        Sets channel types for EEG and resamples both EEG and audio data.

        This method performs the following steps:
        1. Sets channel types for EEG data (e.g., marking specific channels as EOG)
        2. Resamples EEG data to the specified sampling frequency
        3. Resamples Audio data to the specified sampling frequency
        """
        printSectionHeader(f"{Fore.YELLOW}🔧 Setting up data for EEG and Audio")
        
        startTime = time.time()
        
        print('Setting channel types...'.ljust(30), end='')
        channelTypes = {'Fp1':'eog', 'Fp2':'eog'}
        self.eegData.set_channel_types(channelTypes)
        print('✅ Done')
        
        print(f'Resampling EEG data...{self.eegSamplingFrequency}'.ljust(30), end='')
        eegResampleStart = time.time()
        self.eegData.resample(self.eegSamplingFrequency)
        eegResampleTime = time.time() - eegResampleStart
        print(f'✅ Done in {eegResampleTime:.2f} seconds')
        
        print(f"{Fore.CYAN}📊 EEG data shape: {self.eegData.shape}")
        print(f"{Fore.CYAN}🎵 Audio data shape: {self.audioData.shape}")
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ Setup data completed in {totalTime:.2f} seconds")

    def printInfo(self):
        """
        Displays information about the loaded data including sampling frequencies,
        identifiers, and file paths.

        This method prints the following information:
        1. EEG and Audio sampling frequencies
        2. Subject, Session, and Run IDs
        3. Task Name
        4. BIDS Path
        5. Destination Directory
        6. File Name
        """
        printSectionHeader(f"{Fore.MAGENTA}ℹ️  Data Information")
        print(f"{Fore.CYAN}📊 EEG Data Shape: {self.eegData.shape}")
        print(f"{Fore.CYAN}🎵 Audio Data Shape: {self.audioData.shape}")
        print(f"{Fore.YELLOW}⏱️  EEG Sample Rate: {self.eegSamplingFrequency} Hz")
        print(f"{Fore.YELLOW}⏱️  Audio Sample Rate: {self.audioSamplingFrequency} Hz")
        print(f'👤 Subject ID:               {self.subjectId}')
        print(f'🔢 Session ID:               {self.sessionId}')
        print(f'🏃 Run ID:                   {self.runId}')
        print(f'📝 Task Name:                {self.taskName}')
        print(f'🗂️  BIDS Path:                {self.bidsPath}')
        print(f'📁 Destination Directory:    {self.destinationDir}')
        print(f'📄 File Name:                {self.fileName}')
        print(f"{Fore.MAGENTA}{'*' * 60}{Style.RESET_ALL}")

    def createAudio(self):
        """
        Extracts audio data from the XDF file and saves it as a WAV file.
        Reports the time taken for extraction and writing.

        This method performs the following steps:
        1. Creates the destination directory for the audio file
        2. Extracts audio data from the loaded XDF file and flattens the audio data
        3. Normalizes the audio data and converts it to 16-bit PCM format
        4. Writes the audio data to a WAV file
        
        """
        printSectionHeader(f"{Fore.BLUE}🎵 Creating Audio File")
        
        startTime = time.time()
        
        destinationDir = self.destinationDir / 'audio'
        self.ensureDirectoryExists(destinationDir)
        
        print('Extracting audio data...'.ljust(30), end='')
        extractStart = time.time()
        audioData = self.audioData.get_data()
        audioData = audioData.flatten()
        
        # Normalize audio data to [-1, 1] range
        audioData = audioData / np.max(np.abs(audioData))
        
        # Convert to 16-bit PCM
        audioData = (audioData * 32767).astype(np.int16)
        
        extractTime = time.time() - extractStart
        print(f'✅ Done in {extractTime:.2f} seconds')
        
        destinationPath = destinationDir / f'{self.fileName}_audio.wav'
        print(f"{Fore.CYAN}💾 Saving audio file to: {destinationPath}")
        print('Writing audio file...'.ljust(30), end='')
        writeStart = time.time()
        write(str(destinationPath), self.audioSamplingFrequency, audioData)
        writeTime = time.time() - writeStart
        print(f'✅ Done in {writeTime:.2f} seconds')
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ Audio File Created Successfully in {totalTime:.2f} seconds")
        

    def ensureDirectoryExists(self, path):
        """
        Creates the specified directory if it doesn't already exist.

        Parameters:
        - path: Path object representing the directory to be created

        This method uses os.makedirs with the exist_ok flag set to True,
        which creates the directory if it doesn't exist and does nothing
        if it already exists.
        """
        os.makedirs(path, exist_ok=True)

    def createEventsFileForAudio(self):
        """
        Creates a TSV file containing event information for the audio data.
        Includes onset times, durations, and descriptions of events.
        Reports the time taken to write the file.

        This method performs the following steps:
        1. Determines the file name and path for the events file
        2. Creates the destination directory if it doesn't exist
        3. Extracts annotations from the audio data
        4. Writes event information (onset, duration, description) to a TSV file
        5. Measures and reports the time taken to write the file
        6. Returns the path of the created events file

        Returns:
        - Path: The path of the created events file
        """
        printSectionHeader(f"{Fore.YELLOW}📝 Creating Events File for Audio")
        
        startTime = time.time()
        
        fileName = f'{self.fileName}_events.tsv'
        destinationDir = self.destinationDir / 'audio'
        self.ensureDirectoryExists(destinationDir)
        fileNameWithPath = destinationDir / fileName
        
        print('Writing events to file...'.ljust(30), end='')
        writeStart = time.time()
        annotations = self.audioData.annotations
        with open(fileNameWithPath, "w", newline="") as tsvFile:
            writer = csv.writer(tsvFile,  delimiter='\t')
            writer.writerow(['onset', 'duration', 'description'])
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, 
                    annotations.description
                ):
                writer.writerow([onset, duration, description])
        writeTime = time.time() - writeStart
        print(f"{Fore.CYAN}💾 Saving events file to: {fileNameWithPath}")
        print(f'✅ Done in {writeTime:.2f} seconds')
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ Events File Created Successfully in {totalTime:.2f} seconds")
        
        return fileNameWithPath

    def createEDFFile(self):
        """
        Creates a BIDS-compatible EDF file from the EEG data.
        Uses MNE library to write the data in EDF format.
        Reports the time taken to create the file.

        This method performs the following steps:
        1. Uses MNE's write_raw_bids function to create an EDF file
        2. Sets the file format to EDF
        3. Allows preloading of data for faster processing
        4. Overwrites existing files if necessary
        5. Measures and reports the time taken to create the EDF file

        The resulting EDF file is BIDS-compatible and contains the EEG data
        along with necessary metadata.
        """
        printSectionHeader(f"{Fore.CYAN}🧠 Creating BIDS EDF File")
        
        startTime = time.time()
        
        print('Writing EEG data to EDF...'.ljust(30), end='')
        writeStart = time.time()
        write_raw_bids(self.eegData, bids_path=self.bidsPath, allow_preload=True, format='EDF', overwrite=True)
        writeTime = time.time() - writeStart
        print(f'✅ Done in {writeTime:.2f} seconds')
        
        print(f"{Fore.YELLOW}📊 Number of EEG channels: {len(self.eegData.ch_names)}")
        print(f"{Fore.YELLOW}⏱️  EEG duration: {self.eegData.n_times / self.eegData.info['sfreq']} seconds")
        edfFilePath = self.bidsPath.fpath
        print(f"{Fore.CYAN}💾 Saving EDF file to: {edfFilePath}")
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ BIDS EDF File Created Successfully in {totalTime:.2f} seconds")





