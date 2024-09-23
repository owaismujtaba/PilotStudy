import pdb
from src.bids_dataset import XDFData
import src.config as config
from src.data_extractor import extractWordSyllableDataForAllSubjects
from src.dataset_loader import VowelDataset
from src.models import RandomForestModel
from src.trainer import ModelTrainer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

filepath='C:\\DeepRESTORE\\PilotStudy\\RawData\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

subjectId = '01'
sessionId = '01'
runId = '01'


if config.trainModels:
    print('\n' + '=' * 60)
    print('🚀  Starting Model Training Process  🚀'.center(60))
    print('=' * 60)

    vowelDatasetExtractor = VowelDataset(
        rootDir=config.dataDir,
        subjectId=subjectId,
        sessionId=sessionId,
        speechType='Silent',
        languageElement='Experiment',
        eventType='Start',
        trialPhase=None,
        presentationMode='Speech'
    )
    data, labels = vowelDatasetExtractor.vowelData
    data = data[:,:,500:]
    
    # Reshape data for PCA
    nSamples, nChannels, nTimepoints = data.shape
    data_reshaped = data.reshape(nSamples, -1)
    
    # Apply StandardScaler
    print("Applying StandardScaler")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # Apply PCA
    print("Applying PCA")
    pca = PCA(n_components=0.99, svd_solver='full')
    dataPca = pca.fit_transform(data_scaled)
    
    print(f"Original shape: {data.shape}, PCA shape: {dataPca.shape}")
    print(f"Number of components: {pca.n_components_}")
    
    model = RandomForestModel()
    trainer = ModelTrainer()
    trainer.trainModel(model, dataPca, labels)
    
    print('\n' + '*' * 60)
    print('✅  Model Training Process Complete  ✅'.center(60))
    print('*' * 60 + '\n')
    
    pdb.set_trace()



if config.extractSyllableWordData:
    print('\n' + '=' * 60)
    print('🔄  Starting Syllable and Word Data Extraction  🔄'.center(60))
    print('=' * 60)

    extractWordSyllableDataForAllSubjects(
        speechType='Silent',
        languageElement='Experiment',
        eventType='Start',
        trialPhase=None,
        presentationMode='Speech'
    )

    print('\n' + '*' * 60)
    print('✅  Syllable and Word Data Extraction Complete  ✅'.center(60))
    print('*' * 60 + '\n')

if config.createBIDSFile:
    print('\n' + '=' * 60)
    print('📂  Starting BIDS File Creation  📂'.center(60))
    print('=' * 60)

    data = XDFData(
        filePath=filepath,
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId
    )

    print('\n' + '*' * 60)
    print('✅  BIDS File Creation Complete  ✅'.center(60))
    print('*' * 60 + '\n')

