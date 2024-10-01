import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import src.config as config
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam
import pdb
import os

from tensorflow import keras

from imblearn.over_sampling import RandomOverSampler


import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if config.device == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ComplexDualInputNeuralNetwork():
    def __init__(self, timesteps, num_classes):
        super(ComplexDualInputNeuralNetwork, self).__init__()
        
        # Input layers
        input1 = Input(shape=(64, timesteps))
        input2 = Input(shape=(64, 10, timesteps))
        
        # Process first input
        x1 = self.process_1d_input(input1)
        
        # Process second input
        x2 = self.process_2d_input(input2)
        
        # Concatenate both inputs
        combined = layers.Concatenate()([x1, x2])
        
        # Dense layers
        x = self.dense_block(combined, 512)
        x = self.dense_block(x, 256)
        x = self.dense_block(x, 128)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=[input1, input2], outputs=output)
        
    

    def process_1d_input(self, x):
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        return x

    def process_2d_input(self, x):
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        return x

    def dense_block(self, x, units):
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        return x

    def train(self, X, labels, test_size=0.2, epochs=100, batch_size=32):
        # Split the data
        data1, data2 = X
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
            data1, data2, labels, test_size=test_size, random_state=42
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        history = self.model.fit(
            [x1_train, x2_train], y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.15,
            callbacks=callbacks
        )
        predictions = self.model.predict([x1_test, x2_test])
        predictions = np.argmax(predictions, axis=1)
        report = classification_report(predictions, y_test)
        print(report) 
        return history, (x1_test, x2_test, y_test)

    def predict(self, x1_test, x2_test):
        return self.model.predict([x1_test, x2_test])

    def evaluate(self, x1_test, x2_test, y_test):
        return self.model.evaluate([x1_test, x2_test], y_test)


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




class NNModel:
    def __init__(self, inputShape) -> None:
        self.inputShape = inputShape
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.inputShape),  # First layer
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Flatten(),                       
            layers.Dense(config.nClasses, activation='softmax')            # Output layer
        ])
        print(f'Num of classes :{config.nClasses}')
    def train(self, xTrain, yTrain, epochs=100, batchSize=8):
        '''
        from imblearn.over_sampling import SMOTE
        oversampler = SMOTE(random_state=42)
        xResampled, yResampled  = oversampler.fit_resample(xTrain, yTrain)
        '''
                # Split into training and validation sets
        xTrain, xVal, yTrain, yVal = train_test_split(
            xTrain, yTrain, test_size=0.15, random_state=42
        )
        # Early stopping to prevent overfitting
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduceLearningRate = ReduceLROnPlateau(factor=0.5, patience=5)
        # Compile the model with accuracy metric
        self.model.compile(optimizer=Adam(learning_rate=1e-4),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())
        # Train the model
        self.model.fit(
            xTrain, yTrain, 
            validation_data=(xVal, yVal), 
            epochs=epochs, batch_size=batchSize, 
            callbacks=[earlyStopping, reduceLearningRate]
        )

    def predict(self, xTest):
        return self.model.predict(xTest)

  
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

    
