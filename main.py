import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from colorama import Fore, Style, init

from src.models.models import RandomForestModel,  EEGNet
from src.models.trainer import ModelTrainer
from src.dataset.data_extractor import VowelDataExtractor
import src.utils.config as config
from src.utils.utils import printSectionFooter, printSectionHeader
from src.utils.utils import normalize_data

import pdb



init(autoreset=True)

if config.extractEpochsData:
    from src.dataset.data_extractor import extractEpochsDataForAllSubjects
    extractEpochsDataForAllSubjects()

if config.createBIDSFile:
    from src.dataset.bids_dataset import createBIDSDataset
    filePath = '/home/owaismujtaba/projects/PilotStudy/files.csv'
    createBIDSDataset(csvFilePath=filePath)


if config.visualization:
    from src.utils.visualization import plotVowelActivityAllSubjects
    from src.dataset.data_extractor import VowelDataExtractor
    from src.utils.visualization import plotRealSilentAverageActivityAllSubjects

    plotRealSilentAverageActivityAllSubjects()
    #VowelDataExtractor(subjectId='01', sessionId='01')=

    #plotVowelActivityAllSubjects()


if config.trainModels:
    from src.models.models import DenseModel
    from src.dataset.data_extractor import VowelDataExtractor
    printSectionHeader('  Starting Model Training Process  '.center(60))

    subjectId = '01'
    sessionId = '01'
    
    speechType= config.SPEECH_TYPE
    languageElement=config.LANGUAGE_ELEMENT
    eventType=config.EVENT_TYPE
    trialPhase=config.TRIAL_PHASE
    presentationMode=config.PRESENTATION_MODE
    
    
    dataExtractor = VowelDataExtractor(subjectId, sessionId)
    data = dataExtractor.data
    labels = dataExtractor.labels
    
    print(f"{Fore.YELLOW}1️⃣ Applying zscore normalization...{Style.RESET_ALL}")
    
    dataNormalized = normalize_data(data)
       
    nSamples, channels, nTimepoints = data.shape
        
        

    '''
        print(f"{Fore.YELLOW}2️⃣ Applying PCA...{Style.RESET_ALL}")   
        pca = PCA(n_components=0.99, svd_solver='full')
        dataPca = pca.fit_transform(data_scaled)
        
    '''
        #printSectionHeader(f"Original shape: {dataScaled.shape}")
    name = f'sub-{subjectId}_ses-{sessionId}'
    
    destination = Path(
            config.RESULTS_DIR, 
            f'sub-{subjectId}', f'ses-{sessionId}',
            f'{speechType}{languageElement}{eventType}{trialPhase}{presentationMode}'
        )
    
    os.makedirs(destination, exist_ok=True)

    data = np.mean(data, axis=1)
    #model = EEGNet(num_classes=5, channels=data.shape[1], timestamps=data.shape[2])
    model = DenseModel()
    trainer = ModelTrainer(name=name, destination=destination)
    labels = np.array(labels)
    trainer.trainModel(model, data, labels)
        
    print('\n' + '*' * 60)
    print('✅  Model Training Process Complete  ✅'.center(60))
    print('*' * 60 + '\n')


    
