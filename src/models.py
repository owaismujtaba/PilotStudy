import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import src.config as config

class RandomForestModel:
    def __init__(self, 
            nEstimators=[100, 200, 300, 400, 500], randomState=42,
            maxDepth= [None, 10, 20, 30, 40, 50],            
            minSamplesSplit=[2, 5, 10], minSamplesLeaf= [1, 2, 4], 
            maxFeatures=[ 'sqrt', 'log2']):
        self.model = None
        self.nEstimators = nEstimators
        self.randomState = randomState
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxFeatures = maxFeatures

    def train(self, xTrain, yTrain):
        print(f"Original xTrain shape: {xTrain.shape}")
        #xTrain = xTrain.reshape(xTrain.shape[0], -1)
        print(f"Reshaped xTrain shape: {xTrain.shape}")
        print(f"yTrain shape: {yTrain.shape}")
        
        # Define hyperparameter search space
        paramGrid = {
            'n_estimators': self.nEstimators ,
            'max_depth': self.maxDepth,
            'min_samples_split': self.minSamplesSplit,
            'min_samples_leaf':self.minSamplesLeaf,
            'max_features':  self.maxFeatures
        }

        # Create base model
        rfModel = RandomForestClassifier(random_state=self.randomState)

        # Perform randomized search for hyperparameter tuning
        self.model = RandomizedSearchCV(
            estimator=rfModel,
            param_distributions=paramGrid,
            n_iter=100,
            cv=5,
            verbose=1,
            random_state=self.randomState,
            n_jobs=config.nJobs,
            
        )

        # Fit the model
        self.model.fit(xTrain, yTrain)

        # Print details of each fitted model
        for i, model in enumerate(self.model.cv_results_['params']):
            print(f"\nModel {i+1}:")
            print(f"Parameters: {model}")
            print(f"Mean test score: {self.model.cv_results_['mean_test_score'][i]:.4f}")
            print(f"Rank: {self.model.cv_results_['rank_test_score'][i]}")

        # Print the best model details
        print("\nBest Model:")
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best score: {self.model.best_score_:.4f}")

    def predict(self, xTest):
        print(f"Original xTest shape: {xTest.shape}")
        #xTest = xTest.reshape(xTest.shape[0], -1)
        print(f"Reshaped xTest shape: {xTest.shape}")
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(xTest)

    def getBestParams(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.best_params_

    def getFeatureImportance(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.best_estimator_.feature_importances_
