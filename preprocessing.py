import os
import librosa
import math
import json
import numpy as np

dataset= "C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\dataset"
json_file="data.json"
sample_rate = 22050

audio, sr =librosa.load("C:\\Users\\shehr\\Documents\\Machine Learning\\ML_Project\\dataset\\ayat-7\\60_verse7.wav",sr=22050)
mfcc = librosa.feature.mfcc(y=audio,sr=sr,n_fft=2048,n_mfcc=13)
arr = np.zeros((mfcc.shape[0], mfcc.shape[1]))
shape0 = mfcc.shape[0]
shape1=mfcc.shape[1]
duration = librosa.get_duration(y=audio,sr=sr)
samples_per_track=(sample_rate*duration)


def save_mfcc(dataset, json_file, n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    
    data = {
        "mapping": [], #save mappings like possible classifications
        "mfcc": [], #number of coefficients for each mapping (input)
        "labels": [] #labels for each mapping (output)
    }   
    max=00
    k=0
    num_samples_per_segments = int(samples_per_track/num_segments)
    expected_mfcc_vectors_per_segment = math.ceil(num_samples_per_segments/hop_length)
    print(expected_mfcc_vectors_per_segment)
    expected_mfcc_vectors_per_segment=1499
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset)): #loop thru all data
        
       
        if dirpath is not dataset: #this will not let me work in dataset2 root folder
            #store all folder names in data['mapping']
            dirpath_componenets = dirpath.split("\\") 
            label=dirpath_componenets[-1]
            data["mapping"].append(label)
            print("\nProcessing {}".format(label))
            #loop thru all files and load their signals
            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, sr =librosa.load(file_path,sr=sample_rate)


                mfcc = librosa.feature.mfcc(y=signal,
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length
                                                )
                
                arr = np.zeros((shape0, shape1))
                arr[:, :mfcc.shape[1]] = mfcc
                mfcc = arr
                    #audio should be in expected shape so we use mfcc vectors   
                mfcc=mfcc.T

                    #we only store mfcc if it has expected length 
                if len(mfcc)==expected_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)

   

    with open(json_file,"w") as fp:
        json.dump(data, fp, indent=4)


if __name__=="__main__":
    save_mfcc(dataset,json_file,num_segments=5)