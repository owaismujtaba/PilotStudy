from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils import printSectionFooter, printSectionHeader
import pdb
from pathlib import Path
import pandas as pd

class ModelTrainer:
    def __init__(self,name,destination, validationSize=0.15, randomState=42):
        printSectionHeader(" Initializing ModelTrainer ")
        self.name = name
        self.destination=destination
        self.classificationReport = None
        self.validationSize = validationSize
        self.randomState = randomState
        
        printSectionFooter("✅ ModelTrainer Initialization Complete ✅")

    def trainModel(self, model, X, y, name):
        printSectionHeader("易 Training Model 易")
        
        history = model.train(X, y)
        
        modelpath = Path(self.destination, f'{self.name}.h5')
        historypath = Path(self.destination, 'history.csv')
        history.history.to_csv(historypath)
        model.save(modelpath)
        printSectionFooter("✅ Model Training Complete ✅")

    def getClassificationReport(self):
        return self.classificationReport