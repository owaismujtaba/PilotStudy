import pdb
from src.bids_dataset import XDFData
import src.config as config
from src.data_extractor import extractWordSyllableDataForAllSubjects
from src.dataset_loader import VowelDataset
from src.models import RandomForestModel
from src.trainer import ModelTrainer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

filepath='RawData/EEG/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf'

subjectId = '01'
sessionId = '01'
runId = '01'


if config.trainModels:
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
    pdb.set_trace()



if config.extractSyllableWordData:
    extractWordSyllableDataForAllSubjects(
        speechType='Silent',
        languageElement='Experiment',
        eventType='Start',
        trialPhase=None,
        presentationMode='Speech'
    )

if config.createBIDSFile:
    data = XDFData(
        filepath=filepath,
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId
    )

