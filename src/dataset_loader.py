import os
from pathlib import Path
import pdb
import numpy as np
import src.config as config


class VowelDataset:
    def __init__(self, 
            rootDir=config.dataDir, subjectId='01', sessionId='01',
            speechType=None, languageElement=None,
            eventType='Start', trialPhase=None, presentationMode=None
        ):
        print('\n' + '=' * 60)
        print('🚀 Initializing VowelDataset 🚀'.center(60))
        print('=' * 60 + '\n')

        self.subjectId = subjectId

        self.sessionId = sessionId
        self.dataDir = Path(rootDir, 'sub-'+self.subjectId, 'ses-'+self.sessionId)
        self.dataCategory = f"{speechType}_{languageElement}_{eventType}_{trialPhase}_{presentationMode}"
        self.dataDir = Path(self.dataDir, self.dataCategory)
        if not self.dataDir.exists():
            print(f"[ERROR] The directory {self.dataDir} does not exist. Please check the path and try again.")
            return 
        self.syllableDataDir = Path(self.dataDir, 'Syllables')

        self.syllableFilepaths = self.getNumpyFilepaths(self.syllableDataDir)
        

        self.vowelCategories = ['a', 'e', 'i', 'o', 'u']
        self.categorizedSyllablePaths = self.categorizeSyllables()
        self.vowelData = self.loadVowelData()

        print('\n' + '=' * 60)
        print('✅  VowelDataset Initialization Complete  ✅'.center(60))
        print('=' * 60 + '\n')

    def categorizeSyllables(self):
        print('\n' + '=' * 60)
        print('📂 Loading paths for categorized Vowel syllables 📂'.center(60))
        print('=' * 60 + '\n')

        categorized = {vowel: [] for vowel in self.vowelCategories}
        for filepath in self.syllableFilepaths:
            filename = os.path.basename(filepath)
            syllable = os.path.splitext(filename)[0]  # Remove the .npy extension
            for vowel in self.vowelCategories:
                if syllable.endswith(vowel) or syllable.endswith(vowel.upper()):
                    categorized[vowel].append(filepath)
                    break

        print('\n' + '=' * 60)
        print('✅  Categorization Complete  ✅'.center(60))
        print('=' * 60 + '\n')
        return categorized


    def getNumpyFilepaths(self, dataDir):
        print('\n' + '=' * 60)
        print(f'🔍 Scanning directory {dataDir} for .npy files 🔍'.center(60))
        print('=' * 60 + '\n')

        numpyFiles = []

        for root, dirs, files in os.walk(dataDir):
            for file in files:
                if file.endswith('.npy'):
                    numpyFiles.append(os.path.join(root, file))

        print('\n' + '=' * 60)
        print(f'✅  Found {len(numpyFiles)} .npy files  ✅'.center(60))
        print('=' * 60 + '\n')
        return numpyFiles




    def loadVowelData(self):
        print('\n' + '=' * 60)
        print('📊 Loading vowel data from categorized syllable paths 📊'.center(60))
        print('=' * 60 + '\n')

        data, labels = [], []
        for key, paths in self.categorizedSyllablePaths.items():
            print(f'🔠 Extracting data for Vowel: {key}')
            for path in paths:
                syllableData = np.load(path)
                data.append(syllableData)
                labels += [key]*len(syllableData)

        data = np.concatenate(data, axis=0)
        labels = np.array(labels)

        print('\n' + '=' * 60)
        print('✅  Vowel Data Loading Complete  ✅'.center(60))
        print('=' * 60 + '\n')
        return data, labels



