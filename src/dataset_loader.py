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
        self.subjectId = subjectId

        self.sessionId = sessionId
        self.dataDir = Path(rootDir, 'sub-'+self.subjectId, 'ses-'+self.sessionId)
        self.dataCategory = f"{speechType}_{languageElement}_{eventType}_{trialPhase}_{presentationMode}"
        self.dataDir = Path(self.dataDir, self.dataCategory)
        if not self.dataDir.exists():
            print(f"The directory {self.dataDir} does not exist")
            return 
        self.syllableDataDir = Path(self.dataDir, 'Syllables')

        self.syllableFilepaths = self.getNumpyFilepaths(self.syllableDataDir)
        

        self.vowelCategories = ['a', 'e', 'i', 'o', 'u']
        self.categorizedSyllablePaths = self.categorizeSyllables()
        self.vowelData = self.loadVowelData()

    def categorizeSyllables(self):
        print("Loading paths for categorized Vowel syllables")
        categorized = {vowel: [] for vowel in self.vowelCategories}
        for filepath in self.syllableFilepaths:
            filename = os.path.basename(filepath)
            syllable = os.path.splitext(filename)[0]  # Remove the .npy extension
            for vowel in self.vowelCategories:
                if syllable.endswith(vowel) or syllable.endswith(vowel.upper()):
                    categorized[vowel].append(filepath)
                    break

        return categorized


    def getNumpyFilepaths(self, dataDir):
        numpyFiles = []

        for root, dirs, files in os.walk(dataDir):
            for file in files:
                if file.endswith('.npy'):
                    numpyFiles.append(os.path.join(root, file))
        return numpyFiles




    def loadVowelData(self):
        data, labels = [], []
        for key, paths in self.categorizedSyllablePaths.items():
            print("Extracting data for Vowel:", key)
            for path in paths:
                syllableData = np.load(path)
                data.append(syllableData)
                labels += [key]*len(syllableData)

        data = np.concatenate(data, axis=0)
        labels = np.array(labels)

        return data, labels



