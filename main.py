import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from colorama import Fore, Style, init

from src.models.models import RandomForestModel, NNModel, DualInputNeuralNetwork
from src.models.trainer import ModelTrainer
from src.data.bids_dataset import createBIDSDataset
from src.data.data_extractor import VowelDataExtractor
import src.utils.config as config
from src.utils.utils import printSectionFooter, printSectionHeader
from src.utils.utils import normalize_data

import pdb

init(autoreset=True)


if config.visualization:
    from src.utils.visualization import plotVowelActivityAllSubjects

    plotVowelActivityAllSubjects()





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
    filePath = '/home/owaismujtaba/projects/PilotStudy/files.csv'
    createBIDSDataset(csvFilePath=filePath)


    