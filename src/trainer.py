from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils import printSectionFooter, printSectionHeader

class ModelTrainer:
    def __init__(self, validationSize=0.2, randomState=42):
        printSectionHeader("🚀 Initializing ModelTrainer 🚀")
        
        self.classificationReport = None
        self.validationSize = validationSize
        self.randomState = randomState
        
        printSectionFooter("✅ ModelTrainer Initialization Complete ✅")

    def trainModel(self, model, X, y):
        printSectionHeader("🧠 Training Model 🧠")
        
        # Split the data into train and validation sets
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=self.validationSize, random_state=self.randomState
        )

        model.train(xTrain, yTrain)
        yPred = model.predict(xVal)
        self.classificationReport = classification_report(yVal, yPred)
        print(self.classificationReport)
        
        printSectionFooter("✅ Model Training Complete ✅")

    def getClassificationReport(self):
        return self.classificationReport