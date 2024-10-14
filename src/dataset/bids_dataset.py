import os
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from colorama import Fore, Style, init

from mne_bids import BIDSPath, write_raw_bids
from pyxdf import resolve_streams, match_streaminfos
from mnelab.io.xdf import read_raw_xdf
from scipy.io.wavfile import write

from src.utils.utils import printSectionHeader, printSectionFooter
import src.utils.config as config
import pdb



# Initialize colorama for cross-platform color support

init(autoreset=True)

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
            subjectId=None, sessionId=None, 
            runId='01', taskName='PilotStudy'
        ) -> None:
        """
        Initializes the XDFData object with file path and identifiers.
        Sets up paths and calls methods to process the data.

        Parameters:
        - filePath: Path to the XDF file containing EEG and audio data
        - subjectId: Unique identifier for the subject (default: None)
        - sessionId: Identifier for the recording session (default: None)
        - runId: Identifier for the specific run within a session (default: '01')
        - taskName: Name of the task performed during recording (default: 'PilotStudy')

        Raises:
        - ValueError: If filePath, subjectId, or sessionId is None
        """
        printSectionHeader('  Initializing XDFData  ')
        
        # Check if required parameters are not None
        if filePath is None or subjectId is None or sessionId is None:
            raise ValueError("filePath, subjectId, and sessionId must not be None")
        
        self.filePath = filePath
        self.subjectId = subjectId
        self.sessionId = sessionId
        self.runId = runId
        self.taskName = taskName
        self.rawData = None
        self.eegSamplingFrequency = 1000
        self.audioSamplingFrequency = 48000
        self.destinationDir = Path(f'{config.BIDS_DIR}/sub-{self.subjectId}/ses-{self.sessionId}')
        self.fileName = f'sub-{self.subjectId}_ses-{self.sessionId}_task-{self.taskName}_run-{self.runId}'
        
        self.bidsPath = BIDSPath(
            subject=self.subjectId, session=self.sessionId,
            task=self.taskName, run=self.runId, datatype='eeg', 
            root=config.BIDS_DIR
        ) 
        printSectionFooter('✅  Initializing Complete  ✅'.center(config.TERMINAL_WIDTH))

         
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
        printSectionHeader(f"{Fore.CYAN}   🔄 Loading XDF Data 🔄   ")
        
        startTime = time.time()
        
        streams = resolve_streams(self.filePath)
        print(f"{Fore.YELLOW}🔍 Found {len(streams)} streams in the XDF file")
        eegStreamId = match_streaminfos(streams, [{'type':'EEG'}])[0]
        audioStreamId = match_streaminfos(streams, [{'type':'Audio'}])[0]
        
        print(f'{Fore.CYAN}📊 Loading EEG stream...'.ljust(config.TERMINAL_WIDTH), end='')
        eegStart = time.time()
        self.eegData = read_raw_xdf(self.filePath, stream_ids=[eegStreamId])
        eegTime = time.time() - eegStart
        print(f'✅ Done in {eegTime:.2f} seconds')
        
        print(f'{Fore.CYAN}🎵 Loading Audio stream...'.ljust(config.TERMINAL_WIDTH), end='')
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
        """
        printSectionHeader(f"{Fore.YELLOW}🔧 Setting up data for EEG and Audio 🔧")
        
        startTime = time.time()
        
        print(f'{Fore.CYAN}🔄 Setting channel types...'.ljust(config.TERMINAL_WIDTH), end='')
        channelTypeStart = time.time()
        channelTypes = {'Fp1':'eog', 'Fp2':'eog'}
        self.eegData.set_channel_types(channelTypes)
        channelTypeTime = time.time() - channelTypeStart
        print(f'✅ Done in {channelTypeTime:.2f} seconds')
        
        print(f'{Fore.CYAN}📊 Resampling EEG data to {self.eegSamplingFrequency} Hz...'.ljust(config.TERMINAL_WIDTH), end='')
        eegResampleStart = time.time()
        self.eegData.resample(self.eegSamplingFrequency)
        eegResampleTime = time.time() - eegResampleStart
        print(f'✅ Done in {eegResampleTime:.2f} seconds')
        
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
        printSectionHeader(f"{Fore.MAGENTA}ℹ️   *** Data Information ***   ℹ️")
        print(f"{Fore.CYAN}📊 EEG Data Shape: {self.eegData.get_data().shape}")
        print(f"{Fore.CYAN}🎵 Audio Data Shape: {self.audioData.get_data().shape}")
        print(f"{Fore.YELLOW}⏱️  EEG Sample Rate: {self.eegSamplingFrequency} Hz")
        print(f"{Fore.YELLOW}⏱️  Audio Sample Rate: {self.audioSamplingFrequency} Hz")
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print(f"👤 {'Subject ID:':<25} {self.subjectId}".center(config.TERMINAL_WIDTH))
        print(f"🔢 {'Session ID:':<25} {self.sessionId}".center(config.TERMINAL_WIDTH))
        print(f"🏃 {'Run ID:':<25} {self.runId}".center(config.TERMINAL_WIDTH))
        print(f"📝 {'Task Name:':<25} {self.taskName}".center(config.TERMINAL_WIDTH))
        print(f"{Style.RESET_ALL}")
        print(f'🗂️  BIDS Path:{self.bidsPath}'.center(config.TERMINAL_WIDTH))
        print(f'📁 Destination Directory:{self.destinationDir}'.center(config.TERMINAL_WIDTH))
        print(f'📄 File Name:{self.fileName}'.center(config.TERMINAL_WIDTH))
        print(f"{Fore.MAGENTA}{'*' * config.TERMINAL_WIDTH}{Style.RESET_ALL}")

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
        printSectionHeader(f"{Fore.BLUE}🎵 Creating Audio File 🎵")
        
        startTime = time.time()
        
        destinationDir = self.destinationDir / 'audio'
        print(f'{Fore.CYAN}📁 Creating destination directory: {destinationDir}')
        dirCreateStart = time.time()
        self.ensureDirectoryExists(destinationDir)
        dirCreateTime = time.time() - dirCreateStart
        print(f'{Fore.GREEN}✅ Directory created in {dirCreateTime:.2f} seconds')
        
        print(f'{Fore.CYAN}🔍 Extracting audio data...'.ljust(config.TERMINAL_WIDTH), end='')
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
        print(f'{Fore.CYAN}📝 Writing audio file...'.ljust(config.TERMINAL_WIDTH), end='')
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

        Returns:
        - Path: The path of the created events file
        """
        printSectionHeader(f"{Fore.YELLOW}📅 Creating Events File for Audio 📅")
        
        startTime = time.time()
        
        fileName = f'{self.fileName}_events.tsv'
        destinationDir = self.destinationDir / 'audio'
        print(f'{Fore.CYAN}📁 Ensuring directory exists: {destinationDir}')
        dirEnsureStart = time.time()
        self.ensureDirectoryExists(destinationDir)
        dirEnsureTime = time.time() - dirEnsureStart
        print(f'{Fore.GREEN}✅ Directory ensured in {dirEnsureTime:.2f} seconds')
        
        fileNameWithPath = destinationDir / fileName
        
        print(f'{Fore.CYAN}🔍 Extracting annotations...'.ljust(config.TERMINAL_WIDTH), end='')
        annotationStart = time.time()
        annotations = self.audioData.annotations
        annotationTime = time.time() - annotationStart
        print(f'✅ Done in {annotationTime:.2f} seconds')
        
        print(f'{Fore.CYAN}📝 Writing events to file...'.ljust(config.TERMINAL_WIDTH), end='')
        writeStart = time.time()
        with open(fileNameWithPath, "w", newline="") as tsvFile:
            writer = csv.writer(tsvFile,  delimiter='\t')
            writer.writerow(['onset', 'duration', 'description'])
            for onset, duration, description in zip(
                    annotations.onset, annotations.duration, 
                    annotations.description
                ):
                writer.writerow([onset, duration, description])
        writeTime = time.time() - writeStart
        print(f'✅ Done in {writeTime:.2f} seconds')
        
        print(f"{Fore.CYAN}💾 Events file saved to: {fileNameWithPath}")
        
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

        The resulting EDF file is BIDS-compatible and contains the EEG data
        along with necessary metadata.
        """
        printSectionHeader(f"{Fore.CYAN}📊 Creating BIDS EDF File 📊")
        
        startTime = time.time()
        uniqueAnnotations = set(self.eegData.annotations.description)
        eventId = {desc: i+1 for i, desc in enumerate(uniqueAnnotations)}
        
        print(f'{Fore.CYAN}📝 Writing EEG data to EDF...'.ljust(config.TERMINAL_WIDTH), end='')
        writeStart = time.time()
        write_raw_bids(
            self.eegData, 
            bids_path=self.bidsPath, 
            allow_preload=True, 
            format='EDF', 
            overwrite=True,
            event_id=eventId
        )
        writeTime = time.time() - writeStart
        print(f'✅ Done in {writeTime:.2f} seconds')
        
        print(f"{Fore.YELLOW}🔢 Number of EEG channels: {len(self.eegData.ch_names)}")
        print(f"{Fore.YELLOW}⏱️  EEG duration: {self.eegData.n_times / self.eegData.info['sfreq']} seconds")
        edfFilePath = self.bidsPath.fpath
        print(f"{Fore.CYAN}💾 Saving EDF file to: {edfFilePath}")
        
        totalTime = time.time() - startTime
        printSectionFooter(f"{Fore.GREEN}✅ BIDS EDF File Created Successfully in {totalTime:.2f} seconds")


def createBIDSDataset(csvFilePath):
    """
    Creates XDFData objects from information stored in a CSV file using pandas.

    Parameters:
    - csvFilePath: 
        Path to the CSV file containing subject ID, session ID, and 
        file path of the xdf files for respective subject and session

    The CSV file should have the following columns:
    - subject: Unique identifier for the subject
    - session: Identifier for the recording session
    - paths: Path to the XDF file

    This function reads the CSV file using pandas and creates an XDFData object for each row.
    """
    printSectionHeader(f"{Fore.CYAN}🏗️  Creating BIDS Dataset from CSV 🏗️")

    try:
        # Read CSV file using pandas
        df = pd.read_csv(csvFilePath)
        
        for _, row in df.iterrows():
            subjectId = str(row['subject']).zfill(2) 
            sessionId = f"0{row['session']}"  
            filePath = Path(row['paths'])

            print(f"{Fore.YELLOW}🔄 Processing: Subject {subjectId}, Session {sessionId}")
            
            try:
                xdfData = XDFData(
                    filePath=filePath,
                    subjectId=subjectId,
                    sessionId=sessionId
                )
                print(f"{Fore.GREEN}✅ Successfully processed: Subject {subjectId}, Session {sessionId}")
                xdfData.printInfo()
            except Exception as e:
                print(f"{Fore.RED}❌ Error processing: Subject {subjectId}, Session {sessionId}")
                print(f"{Fore.RED}❗ Error: {str(e)}")

    except FileNotFoundError:
        print(f"{Fore.RED}❌ CSV file not found: {csvFilePath}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error reading CSV file: {str(e)}")

    printSectionFooter(f"{Fore.GREEN}✅ BIDS Dataset Creation Complete ✅")
