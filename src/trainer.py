from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

class ModelTrainer:
    def __init__(self, validationSize=0.2, randomState=42):
        self.classificationReport = None
        self.validationSize = validationSize
        self.randomState = randomState

    def trainModel(self, model, X, y):
        # Split the data into train and validation sets
        xTrain, xVal, yTrain, yVal = train_test_split(
            X, y, test_size=self.validationSize, random_state=self.randomState
        )

        model.train(xTrain, yTrain)
        yPred = model.predict(xVal)
        self.classificationReport = classification_report(yVal, yPred)
        print(self.classificationReport)
    def getClassificationReport(self):
        return self.classificationReport