import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler


import src.utils.config as config
import pdb


if config.DEVICE == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class DualInputNeuralNetwork(tf.keras.Model):
    def __init__(self, timesteps, num_classes):
        super(DualInputNeuralNetwork, self).__init__()
        
        # Input layers
        input1 = Input(shape=(64, timesteps))
        input2 = Input(shape=(64, 10, timesteps))
        
        # Process first input
        x1 = layers.Flatten()(input1)
        x1 = layers.Dense(128, activation='relu')(x1)
        x1 = layers.Dropout(0.3)(x1)
        
        # Process second input
        x2 = layers.Reshape((64*10, timesteps))(input2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(128, activation='relu')(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Concatenate both inputs
        combined = layers.Concatenate()([x1, x2])
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=[input1, input2], outputs=output)
        
    def call(self, inputs):
        return self.model(inputs)

    def train(self, X, labels, epochs=50, batch_size=32, validation_split=0.2):
        rawFeatures, morletFeatures = X
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        xTrainMorlet, xTestMorlet, xTrainRaw, xTestRaw, yTrain, yTest = train_test_split(
                morletFeatures, rawFeatures,
                labels, test_size=0.2, random_state=42
        ) 
        history = self.model.fit(
            [xTrainRaw, xTrainMorlet], yTrain, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        predictions = self.model.predict([xTestRaw, xTestMorlet])
        predictions = np.argmax(predictions, axis=1)
        report = classification_report(predictions, yTest)
        print(report) 
        self.evaluate([xTestRaw, xTestMorlet], yTest)
        return history



    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y ):
        return self.model.evaluate(X, y)



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



class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        # Reshape the input if necessary
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Print the model details
        print("SVM Model trained with the following parameters:")
        print(f"Kernel: {self.model.kernel}")
        print(f"C: {self.model.C}")
        print(f"Gamma: {self.model.gamma}")

    def predict(self, X_test):
        # Reshape the input if necessary
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Scale the features
        X_test_scaled = self.scaler.transform(X_test)
        
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test):
        # Reshape the input if necessary
        if len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Scale the features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Print classification report
        print(classification_report(y_test, y_pred))
        
        # Return accuracy score
        return self.model.score(X_test_scaled, y_test)

class EEGNet(tf.keras.Model):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        input_layer = Input((self.Chans, self.Samples, 1))
        
        # Block 1
        x = layers.Conv2D(F1, (1, kernLength), padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.DepthwiseConv2D((self.Chans, 1), use_bias=False, depth_multiplier=D)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 4))(x)
        x = layers.Dropout(dropoutRate)(x)
        
        # Block 2
        x = layers.SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 8))(x)
        x = layers.Dropout(dropoutRate)(x)
        
        x = layers.Flatten()(x)
        output_layer = layers.Dense(self.nb_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    def call(self, inputs):
        return self.model(inputs)
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        xTrain, xTest, yTrain, yTest = train_test_split(
            X, y, test_size=0.15, random_state=42  
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint('best_eegnet_model.h5', save_best_only=True)
        ]

        history = self.model.fit(
            xTrain, yTrain, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks
        )
        predictions = self.model.predict()
        predictions = np.argmax(predictions, axis=1)
        report = classification_report(predictions, yTest)
        print(report) 
        self.evaluate(xTest, yTest)
        return history



    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y ):
        return self.model.evaluate(X, y)


