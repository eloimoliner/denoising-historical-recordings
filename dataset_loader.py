from typing import Tuple, Dict
import ast

import tensorflow as tf
import random
import os
import numpy as np
from scipy.fft import fft, ifft
import soundfile as sf
import librosa
import math
import pandas as pd
import scipy as sp
import glob
from tqdm import tqdm

#generator function. It reads the csv file with pandas and loads the largest audio segments from each recording. If extend=False, it will only read the segments with length>length_seg, trim them and yield them with no further processing. Otherwise, if the segment length is inferior, it will extend the length using concatenative synthesis.
def __noise_sample_generator(info_file,fs, length_seq, split):
    head=os.path.split(info_file)[0]
    load_data=pd.read_csv(info_file)
    #split= train, validation, test
    load_data_split=load_data.loc[load_data["split"]==split]
    load_data_split=load_data_split.reset_index(drop=True)
    while True:
        r = list(range(len(load_data_split)))
        if split!="test":
            random.shuffle(r)
        for i in r:
            segments=ast.literal_eval(load_data_split.loc[i,"segments"])
            if split=="test":
                loaded_data, Fs=sf.read(os.path.join(head,load_data_split["recording"].loc[i],load_data_split["largest_segment"].loc[i]))
            else:
                num=np.random.randint(0,len(segments))
                loaded_data, Fs=sf.read(os.path.join(head,load_data_split["recording"].loc[i],segments[num]))

            if fs!=Fs:
                print("wrong fs, resampling...")
                data=librosa.resample(loaded_data, Fs, fs)

            yield __extend_sample_by_repeating(loaded_data,fs,length_seq)

def __extend_sample_by_repeating(data, fs,seq_len):        
    rpm=78
    target_samp=seq_len
    large_data=np.zeros(shape=(target_samp,2))
    
    if len(data)>=target_samp:
        large_data=data[0:target_samp]
        return large_data
    
    bls=(1000*44100)/1000 #hardcoded
    
    window=np.stack((np.hanning(bls) ,np.hanning(bls)), axis=1) 
    window_left=window[0:int(bls/2),:]
    window_right=window[int(bls/2)::,:]
    bls=int(bls/2)
    
    rps=rpm/60
    period=1/rps
    
    period_sam=int(period*fs)
    
    overhead=len(data)%period_sam
    
    if(overhead>bls):
        complete_periods=(len(data)//period_sam)*period_sam
    else:
        complete_periods=(len(data)//period_sam -1)*period_sam
    
    
    a=np.multiply(data[0:bls], window_left)
    b=np.multiply(data[complete_periods:complete_periods+bls], window_right)
    c_1=np.concatenate((data[0:complete_periods,:],b))
    c_2=np.concatenate((a,data[bls:complete_periods,:],b))
    c_3=np.concatenate((a,data[bls::,:]))
    
    large_data[0:complete_periods+bls,:]=c_1
    
    
    pointer=complete_periods
    not_finished=True
    while (not_finished):
        if target_samp>pointer+complete_periods+bls:
            large_data[pointer:pointer+complete_periods+bls] +=c_2
            pointer+=complete_periods
        else: 
            large_data[pointer::]+=c_3[0:(target_samp-pointer)]
            #finish
            not_finished=False

    return large_data
    

def generate_real_recordings_data(path_recordings, fs=44100, seg_len_s=15, stereo=False):

    records_info=os.path.join(path_recordings,"audio_files.txt")
    num_lines = sum(1 for line in open(records_info))
    f = open(records_info,"r")
    #load data record files
    print("Loading record files")
    records=[]
    seg_len=fs*seg_len_s
    pointer=int(fs*5) #starting at second 5 by default
    for i in tqdm(range(num_lines)):
        audio=f.readline() 
        audio=audio[:-1]
        data, fs=sf.read(os.path.join(path_recordings,audio))
        if len(data.shape)>1 and not(stereo):
            data=np.mean(data,axis=1)
        #elif stereo and len(data.shape)==1:
        #    data=np.stack((data, data), axis=1)

        #normalize
        data=data/np.max(np.abs(data))
        segment=data[pointer:pointer+seg_len]
        records.append(segment.astype("float32"))

    return records

def generate_paired_data_test_formal(path_pianos, path_noises, noise_amount="low_snr",num_samples=-1, fs=44100, seg_len_s=5 , extend=True, stereo=False, prenoise=False):

    print(num_samples)
    segments_clean=[]
    segments_noisy=[]
    seg_len=fs*seg_len_s
    noises_info=os.path.join(path_noises,"info.csv")
    np.random.seed(42)
    if noise_amount=="low_snr":
        SNRs=np.random.uniform(2,6,num_samples)
    elif noise_amount=="mid_snr":
        SNRs=np.random.uniform(6,12,num_samples)

    scales=np.random.uniform(-4,0,num_samples)
    #SNRs=[2,6,12] #HARDCODED!!!!
    i=0
    print(path_pianos[0])
    print(seg_len)
    train_samples=glob.glob(os.path.join(path_pianos[0],"*.wav"))
    train_samples=sorted(train_samples)

    if prenoise:
        noise_generator=__noise_sample_generator(noises_info,fs, seg_len+fs, extend, "test") #Adds 1s of silence add the begiing, longer noise
    else:
        noise_generator=__noise_sample_generator(noises_info,fs, seg_len, extend, "test") #this will take care of everything
    #load data clean files
    for file in tqdm(train_samples):  #add [1:5] for testing
        data_clean, samplerate = sf.read(file)
        if samplerate!=fs: 
            print("!!!!WRONG SAMPLE RATe!!!")
        #Stereo to mono
        if len(data_clean.shape)>1 and not(stereo):
            data_clean=np.mean(data_clean,axis=1)
        #elif stereo and len(data_clean.shape)==1:
        #   data_clean=np.stack((data_clean, data_clean), axis=1)
        #normalize
        data_clean=data_clean/np.max(np.abs(data_clean))
        #data_clean_loaded.append(data_clean)
 
        #framify data clean files
 
        #framify  arguments: seg_len, hop_size
        hop_size=int(seg_len)# no overlap
 
        num_frames=np.floor(len(data_clean)/hop_size - seg_len/hop_size +1) 
        print(num_frames)
        if num_frames==0:
            data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*seg_len-len(data_clean)),))), axis=0)
            num_frames=1

        data_not_finished=True
        pointer=0
        while(data_not_finished):
            if i>=num_samples:
                break
            segment=data_clean[pointer:pointer+seg_len]
            pointer=pointer+hop_size
            if pointer+seg_len>len(data_clean):
                data_not_finished=False
            segment=segment.astype('float32')
    
            #SNRs=np.random.uniform(2,20)
            snr=SNRs[i] 
            scale=scales[i]
            #load noise signal
            data_noise= next(noise_generator)
            data_noise=np.mean(data_noise,axis=1)
            #normalize
            data_noise=data_noise/np.max(np.abs(data_noise))
            new_noise=data_noise #if more processing needed, add here
            #load clean data
            #configure sizes
            power_clean=np.var(segment)
            #estimate noise power
            if prenoise:
                power_noise=np.var(new_noise[fs::])
            else:
                power_noise=np.var(new_noise)

            snr = 10.0**(snr/10.0)

            #sum both signals according to snr
            if prenoise:
                segment=np.concatenate((np.zeros(shape=(fs,)),segment),axis=0) #add one second of silence
            summed=segment+np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!

            summed=summed.astype('float32')
            #yield tf.convert_to_tensor(summed), tf.convert_to_tensor(segment)
  
                
            summed=10.0**(scale/10.0) *summed
            segment=10.0**(scale/10.0) *segment
            segments_noisy.append(summed.astype('float32'))
            segments_clean.append(segment.astype('float32'))
            i=i+1

    return segments_noisy, segments_clean

def generate_test_data(path_music, path_noises,num_samples=-1, fs=44100, seg_len_s=5):

    segments_clean=[]
    segments_noisy=[]
    seg_len=fs*seg_len_s
    noises_info=os.path.join(path_noises,"info.csv")
    SNRs=[2,6,12] #HARDCODED!!!!
    for path in path_music:
        print(path)
        train_samples=glob.glob(os.path.join(path,"*.wav"))
        train_samples=sorted(train_samples)

        noise_generator=__noise_sample_generator(noises_info,fs, seg_len, "test") #this will take care of everything
        #load data clean files
        jj=0
        for file in tqdm(train_samples):  #add [1:5] for testing
            data_clean, samplerate = sf.read(file)
            if samplerate!=fs: 
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1:
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
            #data_clean_loaded.append(data_clean)
     
            #framify data clean files
     
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
     
            num_frames=np.floor(len(data_clean)/hop_size - seg_len/hop_size +1) 
            if num_frames==0:
                data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*seg_len-len(data_clean)),))), axis=0)
                num_frames=1

            pointer=0
            segment=data_clean[pointer:pointer+(seg_len-2*fs)]
            segment=segment.astype('float32')
            segment=np.concatenate(( np.zeros(shape=(2*fs,)), segment), axis=0) #I hope its ok
            #segments_clean.append(segment)
        
            for snr in SNRs:
                #load noise signal
                data_noise= next(noise_generator)
                data_noise=np.mean(data_noise,axis=1)
                #normalize
                data_noise=data_noise/np.max(np.abs(data_noise))
                new_noise=data_noise #if more processing needed, add here
                #load clean data
                #configure sizes
                #estimate clean signal power
                power_clean=np.var(segment)
                #estimate noise power
                power_noise=np.var(new_noise)

                snr = 10.0**(snr/10.0)

                #sum both signals according to snr
                summed=segment+np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!
                summed=summed.astype('float32')
                #yield tf.convert_to_tensor(summed), tf.convert_to_tensor(segment)
      
                segments_noisy.append(summed.astype('float32'))
                segments_clean.append(segment.astype('float32'))

    return segments_noisy, segments_clean

def generate_val_data(path_music, path_noises,split,num_samples=-1, fs=44100, seg_len_s=5):

    val_samples=[]
    for path in path_music:
        val_samples.extend(glob.glob(os.path.join(path,"*.wav")))

    #load data clean files
    print("Loading clean files")
    data_clean_loaded=[]
    for ff in tqdm(range(0,len(val_samples))):  #add [1:5] for testing
        data_clean, samplerate = sf.read(val_samples[ff])
        if samplerate!=fs: 
            print("!!!!WRONG SAMPLE RATe!!!")
        #Stereo to mono
        if len(data_clean.shape)>1 :
            data_clean=np.mean(data_clean,axis=1)
        #normalize
        data_clean=data_clean/np.max(np.abs(data_clean))
        data_clean_loaded.append(data_clean)
        del data_clean

    #framify data clean files
    print("Framifying clean files")
    seg_len=fs*seg_len_s
    segments_clean=[]
    for file in tqdm(data_clean_loaded):

        #framify  arguments: seg_len, hop_size
        hop_size=int(seg_len)# no overlap

        num_frames=np.floor(len(file)/hop_size - seg_len/hop_size +1) 
        pointer=0
        for i in range(0,int(num_frames)):
            segment=file[pointer:pointer+seg_len]
            pointer=pointer+hop_size
            segment=segment.astype('float32')
            segments_clean.append(segment)

    del data_clean_loaded
    
    SNRs=np.random.uniform(2,20,len(segments_clean))
    scales=np.random.uniform(-6,4,len(segments_clean))
    #noise_shapes=np.random.randint(0,len(noise_samples), len(segments_clean))
    noises_info=os.path.join(path_noises,"info.csv")

    noise_generator=__noise_sample_generator(noises_info,fs, seg_len,  split) #this will take care of everything
    

    #generate noisy segments
    #load noise samples using pandas dataframe. Each split (train, val, test) should have its unique csv info file

    #noise_samples=glob.glob(os.path.join(path_noises,"*.wav"))
    segments_noisy=[]
    print("Processing noisy segments")

    for i in tqdm(range(0,len(segments_clean))):
        #load noise signal
        data_noise= next(noise_generator)
        #Stereo to mono
        data_noise=np.mean(data_noise,axis=1)
        #normalize
        data_noise=data_noise/np.max(np.abs(data_noise))
        new_noise=data_noise #if more processing needed, add here
        #load clean data
        data_clean=segments_clean[i]
        #configure sizes
        
         
        #estimate clean signal power
        power_clean=np.var(data_clean)
        #estimate noise power
        power_noise=np.var(new_noise)

        snr = 10.0**(SNRs[i]/10.0)

        #sum both signals according to snr
        summed=data_clean+np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!
            #the rest is normal
        
        summed=10.0**(scales[i]/10.0) *summed
        segments_clean[i]=10.0**(scales[i]/10.0) *segments_clean[i]

        segments_noisy.append(summed.astype('float32'))
        
    return segments_noisy, segments_clean

        

def generator_train(path_music, path_noises,split, fs=44100, seg_len_s=5, extend=True, stereo=False):

    train_samples=[]
    for path in path_music:
        train_samples.extend(glob.glob(os.path.join(path.decode("utf-8") ,"*.wav")))

    seg_len=fs*seg_len_s
    noises_info=os.path.join(path_noises.decode("utf-8"),"info.csv")
    noise_generator=__noise_sample_generator(noises_info,fs, seg_len,  split.decode("utf-8")) #this will take care of everything
    #load data clean files
    while True:
        random.shuffle(train_samples)
        for file in train_samples:  
            data, samplerate = sf.read(file)
            if samplerate!=fs: 
                print("!!!!WRONG SAMPLE RATe!!!")
                data=np.transpose(data)
                data=librosa.resample(data, samplerate, 44100)
                data=np.transpose(data)
            data_clean=data
            #Stereo to mono
            if len(data.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)

            #normalize
            data_clean=data_clean/np.max(np.abs(data_clean))
     
            #framify data clean files
     
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)
     
            num_frames=np.floor(len(data_clean)/seg_len) 
            if num_frames==0:
                data_clean=np.concatenate((data_clean, np.zeros(shape=(int(2*seg_len-len(data_clean)),))), axis=0)
                num_frames=1
                pointer=0
                data_clean=np.roll(data_clean, np.random.randint(0,seg_len)) #if only one frame, roll it for augmentation
            elif num_frames>1:
                pointer=np.random.randint(0,hop_size)  #initial shifting, graeat for augmentation, better than overlap as we get different frames at each "while" iteration
            else:
                pointer=0

            data_not_finished=True
            while(data_not_finished):
                segment=data_clean[pointer:pointer+seg_len]
                pointer=pointer+hop_size
                if pointer+seg_len>len(data_clean):
                    data_not_finished=False
                segment=segment.astype('float32')
        
                SNRs=np.random.uniform(2,20)
                scale=np.random.uniform(-6,4)
        
     
                #load noise signal
                data_noise= next(noise_generator)
                data_noise=np.mean(data_noise,axis=1)
                #normalize
                data_noise=data_noise/np.max(np.abs(data_noise))
                new_noise=data_noise #if more processing needed, add here
                #load clean data
                #configure sizes
                if stereo:
                    #estimate clean signal power
                    power_clean=0.5*np.var(segment[:,0])+0.5*np.var(segment[:,1])
                    #estimate noise power
                    power_noise=0.5*np.var(new_noise[:,0])+0.5*np.var(new_noise[:,1])
                else:
                    #estimate clean signal power
                    power_clean=np.var(segment)
                    #estimate noise power
                    power_noise=np.var(new_noise)

                snr = 10.0**(SNRs/10.0)

         
                #sum both signals according to snr
                summed=segment+np.sqrt(power_clean/(snr*power_noise))*new_noise #not sure if this is correct, maybe revisit later!!
                summed=10.0**(scale/10.0) *summed
                segment=10.0**(scale/10.0) *segment
         
                summed=summed.astype('float32')
                yield tf.convert_to_tensor(summed), tf.convert_to_tensor(segment)
        
def load_data(buffer_size, path_music_train, path_music_val,  path_noises,  fs=44100, seg_len_s=5,  extend=True, stereo=False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    print("Generating train dataset")
    trainshape=int(fs*seg_len_s)

    dataset_train = tf.data.Dataset.from_generator(generator_train,args=(path_music_train, path_noises,"train", fs, seg_len_s,  extend, stereo), output_shapes=(tf.TensorShape((trainshape,)),tf.TensorShape((trainshape,))), output_types=(tf.float32, tf.float32) )


    print("Generating validation dataset")
    segments_noisy, segments_clean=generate_val_data(path_music_val, path_noises,"validation",fs=fs, seg_len_s=seg_len_s)
    
    dataset_val=tf.data.Dataset.from_tensor_slices((segments_noisy, segments_clean))

    return dataset_train.shuffle(buffer_size), dataset_val

def load_data_test(buffer_size, path_pianos_test,   path_noises,  **kwargs) -> Tuple[tf.data.Dataset]:
    print("Generating test dataset")
    segments_noisy, segments_clean=generate_test_data(path_pianos_test, path_noises, extend=True, **kwargs)
    dataset_test=tf.data.Dataset.from_tensor_slices((segments_noisy, segments_clean))
    #dataset_test=tf.data.Dataset.from_tensor_slices((segments_noisy[1:3], segments_clean[1:3]))
    #train_dataset = train.cache().shuffle(buffer_size).take(info.splits["train"].num_examples)
    return dataset_test
def load_data_formal( path_pianos_test,   path_noises,  **kwargs) -> Tuple[tf.data.Dataset]:
    print("Generating test dataset")
    segments_noisy, segments_clean=generate_paired_data_test_formal(path_pianos_test, path_noises, extend=True, **kwargs)
    print("segments::")
    print(len(segments_noisy))
    dataset_test=tf.data.Dataset.from_tensor_slices((segments_noisy, segments_clean))
    #dataset_test=tf.data.Dataset.from_tensor_slices((segments_noisy[1:3], segments_clean[1:3]))
    #train_dataset = train.cache().shuffle(buffer_size).take(info.splits["train"].num_examples)
    return dataset_test

def load_real_test_recordings(buffer_size, path_recordings,   **kwargs) -> Tuple[tf.data.Dataset]:
    print("Generating real test dataset")
        
    segments_noisy=generate_real_recordings_data(path_recordings, **kwargs)

    dataset_test=tf.data.Dataset.from_tensor_slices(segments_noisy)
    #train_dataset = train.cache().shuffle(buffer_size).take(info.splits["train"].num_examples)
    return dataset_test
