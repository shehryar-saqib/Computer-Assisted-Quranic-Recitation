import os
import librosa
import json
import numpy as np
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pydub import AudioSegment
import google.cloud.speech as speech
import soundfile


dataset = "data.json"

def load_data(dataset):

    with open(dataset, "r") as f:
        data =  json.load(f)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    Z = np.array(data["mapping"])

    return X,y,Z


def prepare_datasets(test_size,validation_size):

    X,y, z = load_data(dataset)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=validation_size)

    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]


    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = tf.keras.Sequential()

    #2nd
    model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape,padding='same'))
    model.add(tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    
    #2nd
    model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=input_shape,padding='same'))
    model.add(tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    
    #2nd
    model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=input_shape,padding='same'))
    model.add(tf.keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    #3rd
    model.add(tf.keras.layers.Conv2D(64,(2,2),activation="relu",input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((2,2),strides=(2,2),padding='same'))
    model.add(tf.keras.layers.BatchNormalization())


    #flatten layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    #output
    model.add(tf.keras.layers.Dense(7,activation="softmax"))

    return model
    


def predict(model, X, y):
    X=X[np.newaxis,...]
    prediction = model.predict(X)
    index=np.argmax(prediction,axis=1)
    predicted=Z[index]
    print("Expected: {} and predicted: {}".format(y, index+1))
    return predicted

def testingOnOurVoiceAndCorrection(audio_file_path,exp):
    audio, sr =librosa.load("C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\dataset\\ayat-7\\60_verse7.wav",sr=22050)
    mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_fft=2048,n_mfcc=13)
    arr = np.zeros((mfcc.shape[0], mfcc.shape[1]))
    shape0 = mfcc.shape[0]
    shape1=mfcc.shape[1]
    duration = librosa.get_duration(y=audio,sr=sr)
    samples_per_track=(22050*duration)
    signal, sr =librosa.load(audio_file_path,sr=22050)
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sr,
                                n_fft=2048,
                                n_mfcc=13,
                                hop_length=512
                                )
    arr = np.zeros((shape0, shape1))
    arr[:, :mfcc.shape[1]] = mfcc
    mfcc = arr 
    mfcc=mfcc.T
    heavy = predict(reconstructed_model, mfcc,exp)
    print(heavy)
    
    ayat_1='بسم الله الرحمن الرحيم'
    ayat_2='الحمد لله رب العالمين'
    ayat_3='الرحمن الرحيم'
    ayat_4='مالك يوم الدين'
    ayat_5='اياك نعبد واياك نستعين'
    ayat_6='اهدنا الصراط المستقيم'
    ayat_7='صراط الذين انعمت عليهم غير المغضوب عليهم ولا الضالين'

    test=[]
    if(heavy==['ayat-1']):
        ayat_1=ayat_1.strip()
        ayat_1=ayat_1.replace('.','')
        ayat_1=ayat_1.split()
        test=ayat_1

    if(heavy==['ayat-2']):
        ayat_2=ayat_2.strip()
        ayat_2=ayat_2.replace('.','')
        ayat_2=ayat_2.split()
        test=ayat_2

    if(heavy==['ayat-3']):
        ayat_3=ayat_3.strip()
        ayat_3=ayat_3.replace('.','')
        ayat_3=ayat_3.split()
        test=ayat_3

    if(heavy==['ayat-4']):
        ayat_4=ayat_4.strip()
        ayat_4=ayat_4.replace('.','')
        ayat_4=ayat_4.split()
        test=ayat_4

    if(heavy==['ayat-5']):
        ayat_5=ayat_5.strip()
        ayat_5=ayat_5.replace('.','')
        ayat_5=ayat_5.split()
        test=ayat_5

    if(heavy==['ayat-6']):
        ayat_6=ayat_6.strip()
        ayat_6=ayat_6.replace('.','')
        ayat_6=ayat_6.split()
        test=ayat_6

    if(heavy==['ayat-7']):
        ayat_7=ayat_7.strip()
        ayat_7=ayat_7.replace('.','')
        ayat_7=ayat_7.split()
        test=ayat_7


    extracted_features=[]
    client=speech.SpeechClient.from_service_account_file("C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\key.json")
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound.set_channels(1)
    audio_for_wav="C:\\Users\shehr\\Documents\\Machine Learning\\ML_Project\\1\\23.wav"
    sound.export(audio_for_wav, format="wav")
    with open(audio_for_wav,'rb') as f:
        mp3_data=f.read()
    audio_file=speech.RecognitionAudio(content=mp3_data)
    config=speech.RecognitionConfig(
        enable_automatic_punctuation=True,
        language_code='ar',
        audio_channel_count = 1,
        enable_separate_recognition_per_channel=True
        )
    response=client.recognize(
        config=config,
        audio=audio_file,
        )
    for result in response.results:
        transcript=format(result.alternatives[0].transcript)
    extracted_features.append(transcript)

    extracted_features="".join(extracted_features)
    extracted_features=extracted_features.strip()
    extracted_features=extracted_features.replace('.','')
    extracted_features=extracted_features.split()
    print(extracted_features)

# Check if the number of words in each string is the same
    if len(test) != len(extracted_features):
        print("Incorrect Recitation")
    else:
    # Check if the sequence of words is the same
        if test == extracted_features:
            print("Correct Recitation")
        else:    
        # Check if the words are the same, regardless of their order
            set1 = set(test)
            set2 = set(extracted_features)
            if set1 == set2:
                print("The words are recited correctly, but in different order.")
            else:
                for i in range(len(test)):
                    if test[i] != extracted_features[i]:
                        print(f"The words at index {i} are different.")
                        print(f"The word incorrectly recited is:{test[i]}")
     
        

if __name__ == "__main__":

    X , Y , Z =load_data(dataset)
    #create train/validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.15,0.1)

    #build CNN
    # input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    # #print()
    # model = build_model(input_shape)

    #compile the network
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",metrics=['accuracy'])

    #train the network
    # model.fit(X_train, y_train, validation_data=(X_validation,y_validation),batch_size=16,epochs=20)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.000004)
    # model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",metrics=['accuracy'])

    #train the network
    # model.fit(X_train, y_train, validation_data=(X_validation,y_validation),batch_size=16,epochs=15)

    #evaluate the network on test set
    # model.save("C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\my_model.h5")
    
    reconstructed_model=tf.keras.models.load_model("C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\my_model.h5")
    test_error, test_accuracy = reconstructed_model.evaluate(X_test,y_test,verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    testingOnOurVoiceAndCorrection("C:\\Users\shehr\\Documents\\Machine Learning\\ML_Project\\1\\7.wav",7)
 
    
