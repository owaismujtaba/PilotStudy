import mne
import numpy as np
import pyxdf
from datetime import datetime
from collections import Counter
import src.config as confg


class XDFData:
    def __init__(self, filepath=None) -> None:
        self.rawData = None
        self.filepath = filepath
        self.loadXdfData()
        self.setupData()
        self.printInfo()
        self.createMNEObjectForEEG()
        self.makeAnnotations()
    
    def loadXdfData(self):
        print('Loading XDF Data')
        self.data, self.header = pyxdf.load_xdf(self.filepath)
    
    def setupData(self,):
        print('Setting Up XDF Data')
        self.eegChannelNames = []
        self.measDate = self.header['info']['datetime'][0]
        self.markers = self.data[1]['time_series']
        self.markers.pop()
        self.markers = [marker[0] for marker in self.markers]
        self.markersTimestamps = self.data[1]['time_stamps']
        self.eegData = self.data[2]['time_series']
        self.eegSamplingFrequency = int(float(self.data[2]['info']['nominal_srate'][0]))
        self.eegTimestamps = self.data[2]['time_stamps']
        self.audioData = self.data[3]['time_series']
        self.audioTimestamps = self.data[3]['time_stamps']
        self.audioSamplingFrequency = int(float(self.data[3]['info']['nominal_srate'][0]))
        channelNames = self.data[2]['info']['desc'][0]['channels'][0]['channel']
        for item in channelNames:
            self.eegChannelNames.append(item['label'][0]) 
    def printInfo(self):
        print(f'No of Markers: {len(self.markers)} No .of Marker Timestamps: {self.markersTimestamps.shape[0]}')
        print(f'EEG data Shape: {self.eegData.shape} No .of eeg Timestamps: {self.eegTimestamps.shape[0]}')
        print(f'Audio data Shape: {self.audioData.shape} No .of audio Timestamps: {self.audioTimestamps.shape[0]}')
        print('EEG Channels:', self.eegChannelNames)
        print(f'Sampling Frequency ::: EEG: {self.eegSamplingFrequency}, Audio: {self.audioSamplingFrequency}')
    def createMNEObjectForEEG(self):
        print('Creating MNE Data')
        meas_date = datetime.strptime(self.measDate, '%Y-%m-%dT%H:%M:%S%z')
        meas_date = (int(meas_date.timestamp()), int(meas_date.microsecond))
        info = mne.create_info(
            ch_names=self.eegChannelNames, 
            sfreq=self.eegSamplingFrequency, 
            ch_types='eeg',
        )
        info.set_meas_date(meas_date)
        self.rawEegMNEData = mne.io.RawArray(self.eegData.T, info)
    def getOnsetCodesForAnnotations(self):
        print('Setting up Annotation Codes for Data')
        codes = []
        onset = []
        description = []
        duration = []
        markers = self.markers
        markersTimestamps = self.markersTimestamps - self.eegTimestamps[0]
        for index in range(len(markers)-1):
            marker = markers[index]
            code = ''
            if 'Silent' in marker:
                code += '10,' 
            elif 'Real' in marker:
                code += '11,' 
            else:
                code += ','

            if 'Word' in marker:
                code += '12,' 
            elif 'Syllable' in marker:
                code += '13,' 
            else:
                code += ','

            if 'Practice' in marker:
                code += '14,' 
            elif 'Experiment' in marker:
                code += '15,' 
            else:
                code += ','

            if 'Start' in marker:
                code += '16,' 
            elif 'End' in marker:
                code += '17,' 
            else:
                code += ','

            if 'Fixation' in marker:
                code += '18,' 
            elif 'Stimulus' in marker:
                code += '19,' 
            elif 'ISI' in marker:
                code += '20,' 
            elif 'ITI' in marker:
                code += '21,' 
            elif 'Speech' in marker:
                code += '22,' 
            else:
                code += ','
            
            if 'Audio' in marker:
                code += '23,'
            elif 'Text' in marker:
                code += '24,'
            elif 'Pictures' in marker:
                code += '25,'
            else:
                code += ','
            wordOrSyllable = marker.split(':')[1].split('_')[1]

            code += wordOrSyllable
            codes.append(code)
            onset.append(markersTimestamps[index])
            description.append(marker)
            duration.append(markersTimestamps[index+1]-markersTimestamps[index])

        return onset, codes, duration
    def makeAnnotations(self):
        print('Annotating Data')
        onset, codes, duration = self.getOnsetCodesForAnnotations()
        rawMNEWithAnnotations = self.rawEegMNEData.copy()
        rawMNEWithAnnotations.set_annotations(mne.Annotations(onset=onset, description=codes, duration=duration))

        self.rawMNEWithAnnotations = rawMNEWithAnnotations


class SyllableDataProcessor:
    def __init__(self, mneDataObject) -> None:
        self.data = mneDataObject


    def getSyllabelData(self):
        events, eventIds = mne.events_from_annotations(self.data.rawMNEWithAnnotations, verbose=False)
        eventIdsReversed = {str(value): key for key, value in eventIds.items()} 
        codes, eventTimings = [], []
        for event in events:
            eventCode = eventIdsReversed.get(str(event[2]), None)
            if eventCode:
                code = self._getCode(eventCode)
                if code:
                    codes.append(code)
                    eventTimings.append(event[0])
        semanticEvents = np.array([[timing, 0, code] for timing, code in zip(eventTimings, codes)])
        semanticEventIds = {'Silence': 1, 'Real': 2}
    
    def _getCode(eventCode):
        items = eventCode.split(',')
        if items[3] == '16' and items[4] == '19':
            if items[0] == '10':
                return 1
            elif items[0] == '11':
                return 2
            else:
                return None
        else:
            return None