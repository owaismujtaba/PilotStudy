import pdb
from colorama import Fore, Style, init
from scipy.stats import zscore
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import pandas as pd

from data.bids_dataset import XDFData
import utils.config as config
from data.data_extractor import GroupDataExtractor
from src.models import RandomForestModel, NNModel, DualInputNeuralNetwork
from src.trainer import ModelTrainer
from utils.utils import printSectionFooter, printSectionHeader
from utils.utils import normalize_data

init(autoreset=True)


if config.visualization:
    from src.visualization import plot_vowvel_acitvity

    plot_vowvel_acitvity()





if config.trainModels:
    printSectionHeader('  Starting Model Training Process  '.center(60))

    file = pd.read_csv('/home/owaismujtaba/projects/PilotStudy/files.csv')

    subjects = file['subject'].values
    sessions = file['session'].values
    speechType='Silent'
    languageElement='Experiment'
    eventType='Start'
    trialPhase=None
    presentationMode='Speech'
    for index in range(len(sessions)):
        subjectId = subjects[index]
        sessionId = sessions[index]
        
        if subjectId< 10:
            subjectId = f'0{subjectId}'
        else:
            subjectId = str(subjectId)
        sessionId = f'0{sessionId}'

        dataExtractor = GroupDataExtractor(
            subjectId=subjectId,
            sessionId=sessionId,
            taskName='PilotStudy',
            speechType=speechType,
            languageElement=languageElement,
            eventType=eventType,
            trialPhase=trialPhase,
            presentationMode=presentationMode,        
        )

        morletData, rawData, labels = dataExtractor.morletFeatures, dataExtractor.rawFeatures, dataExtractor.labels
        print(f"{Fore.YELLOW}1️⃣ Applying zscore normalization...{Style.RESET_ALL}")
        
        morletData = normalize_data(morletData)
        rawData = normalize_data(rawData)

        nSamples,channels,  nFreqBins, nTimepoints = morletData.shape
        
        

        '''
        print(f"{Fore.YELLOW}2️⃣ Applying PCA...{Style.RESET_ALL}")   
        pca = PCA(n_components=0.99, svd_solver='full')
        dataPca = pca.fit_transform(data_scaled)
        
        '''
        #printSectionHeader(f"Original shape: {dataScaled.shape}")
        name = f'sub-{subjectId}_ses-{sessionId}'
        destination = Path(
            config.resultsDir, speechType,
            languageElement, eventType, trialPhase,presentationMode
        )
        os.makedirs(destination)
        

        model = DualInputNeuralNetwork(timesteps=1501, num_classes=5)
        trainer = ModelTrainer(name=name, destination=destination)
        labels = np.array(labels)
        trainer.trainModel(model, [rawData, morletData], labels)
        
        print('\n' + '*' * 60)
        print('✅  Model Training Process Complete  ✅'.center(60))
        print('*' * 60 + '\n')

if config.loadData:
    printSectionHeader('  Starting Data Extraction  '.center(60))

    import pandas as pd

    file = pd.read_csv('/home/owaismujtaba/projects/PilotStudy/files.csv')

    subjects = file['subject'].values
    sessions = file['session'].values
    for index in range(len(sessions)):
        subjectId = subjects[index]
        sessionId = sessions[index]
        
        if subjectId< 10:
            subjectId = f'0{subjectId}'
        else:
            subjectId = str(subjectId)
        sessionId = f'0{sessionId}'
        data = GroupDataExtractor(
            subjectId=subjectId,
            sessionId=sessionId,
            runId=runId,
            taskName='PilotStudy',
            speechType='Real',
            languageElement='Experiment',
            eventType='Start',
            trialPhase=None,
            presentationMode='Speech'
        )

    printSectionFooter('✅  Data Extraction Complete  ✅')

    printSectionFooter('✅  Syllable and Word Data Extraction Complete  ✅')

if config.createBIDSFile:

    import pandas as pd

    file = pd.read_csv('/home/owaismujtaba/projects/PilotStudy/files.csv')

    subjects = file['subject'].values
    sessions = file['session'].values
    paths = file['paths'].values
    for index in range(len(sessions)):
        subjectId = subjects[index]
        sessionId = sessions[index]
        if subjectId<13:
            continue
        
        if subjectId< 10:
            subjectId = f'0{subjectId}'
        else:
            subjectId = str(subjectId)
        sessionId = f'0{sessionId}'
        filepath = paths[index]


        printSectionHeader('  Starting BIDS File Creation  ')
        printSectionHeader('  XDFData Details  ')
        print(f"File Path: {filepath}".center(60))
        print(f"Subject ID: {subjectId}".center(60))
        print(f"Session ID: {sessionId}".center(60))
        print(f"Run ID: {runId}".center(60))
        print('-' * 60 + '\n')
        data = XDFData(
            filePath=filepath,
            subjectId=subjectId,
            sessionId=sessionId,
            runId=runId
        )

    printSectionFooter('✅  BIDS File Creation Complete  ✅'.center(60))

