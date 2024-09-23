# EEG Data Processing and Analysis Pipeline

## Overview

This project implements a pipeline for processing and analyzing EEG (Electroencephalography) data, specifically focusing on speech and language tasks. The pipeline includes data extraction from BIDS-formatted datasets, preprocessing of EEG signals, and training of machine learning models for classification tasks.

## Tools and Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MNE](https://img.shields.io/badge/MNE-Python-blue?style=for-the-badge)

## Key Components

1. **Data Extractor** (`src/data_extractor.py`)
   - `NeuralDatasetExtractor`: Extracts EEG data from BIDS-formatted datasets.
   - `WordSyllableDataExtractor`: Processes and saves word and syllable data.

2. **Dataset Loader** (`src/dataset_loader.py`)
   - `VowelDataset`: Loads and categorizes vowel data from processed EEG signals.

3. **Model Trainer** (`src/trainer.py`)
   - `ModelTrainer`: Handles the training and evaluation of machine learning models.

4. **Configuration** (`src/config.py`)
   - Contains global configuration settings for the project.

## Setup and Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the BIDS-formatted EEG dataset in the directory specified in `config.py`.

## Usage

1. **Data Extraction**:
   ```python
   from src.data_extractor import extractWordSyllableDataForAllSubjects

   extractWordSyllableDataForAllSubjects(
       speechType='Overt',
       languageElement='Word',
       eventType='Start',
       trialPhase='Stimulus',
       presentationMode='Audio'
   )
   ```

2. **Load Dataset**:
   ```python
   from src.dataset_loader import VowelDataset

   dataset = VowelDataset(
       subjectId='01',
       sessionId='01',
       speechType='Overt',
       languageElement='Word',
       eventType='Start',
       trialPhase='Stimulus',
       presentationMode='Audio'
   )
   ```

3. **Train Model**:
   ```python
   from src.trainer import ModelTrainer
   from your_model_file import YourModel

   X, y = dataset.vowelData
   model = YourModel()
   trainer = ModelTrainer()
   trainer.trainModel(model, X, y)
   ```

## Configuration

Adjust settings in `src/config.py` to match your environment and dataset structure:

- `bidsDir`: Path to the BIDS-formatted dataset
- `dataDir`: Path for storing processed data
- `tmin` and `tmax`: Time window for epoch extraction

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- MNE-Python for EEG data processing
- scikit-learn for machine learning tools