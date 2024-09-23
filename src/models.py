import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import src.config as config
import tensorflow as tf
from tensorflow.keras import layers, models

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

class Block(tf.keras.layers.Layer):
    def __init__(self, inplace):
        super(Block, self).__init__()
        self.conv1 = layers.Conv1D(32, 2, strides=2, padding='valid')
        self.conv2 = layers.Conv1D(32, 4, strides=2, padding='same')
        self.conv3 = layers.Conv1D(32, 8, strides=2, padding='same')
        self.relu = layers.ReLU()

    def call(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x = tf.concat([x1, x2, x3], axis=-1)
        return x

class ChronoNet(tf.keras.Model):
    def __init__(self, channel):
        super(ChronoNet, self).__init__()
        self.block1 = Block(channel)
        self.block2 = Block(96)
        self.block3 = Block(96)
        self.gru1 = layers.GRU(32, return_sequences=True)
        self.gru2 = layers.GRU(32, return_sequences=True)
        self.gru3 = layers.GRU(32, return_sequences=True)
        self.gru4 = layers.GRU(32, return_sequences=True)
        self.gru_linear = layers.Dense(64)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(5)
        self.relu = layers.ReLU()

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        gru_out1 = self.gru1(x)
        gru_out2 = self.gru2(gru_out1)
        gru_out = tf.concat([gru_out1, gru_out2], axis=-1)
        gru_out3 = self.gru3(gru_out)
        gru_out = tf.concat([gru_out1, gru_out2, gru_out3], axis=-1)
        linear_out = self.relu(self.gru_linear(gru_out))
        gru_out4 = self.gru4(linear_out)
        x = self.flatten(gru_out4)
        x = self.fc1(x)
        return x

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test):
        return self.predict(x_test)

    
