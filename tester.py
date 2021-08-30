
import os
import numpy as np
import cv2
import librosa
import imageio
import tensorflow as tf
import soundfile as sf
import subprocess
from tqdm import tqdm
from vggish.vgg_distance import process_wav
import pandas as pd                                    
from scipy.io import loadmat

class Tester():
    def __init__(self, model, path_experiment, args):
        if model !=None:
            self.model=model
            print(self.model.summary())
        self.args=args
        self.path_experiment=path_experiment

    def init_inference(self,  dataset_test=None,num_test_segments=0 , fs=44100, stft_args=None, PEAQ_dir=None, alg_dir=None, PEMOQ_dir=None):

        self.num_test_segments=num_test_segments
        self.dataset_test=dataset_test

        if self.dataset_test!=None:
            self.dataset_test=self.dataset_test.take(self.num_test_segments)

        self.fs=fs
        self.stft_args=stft_args
        self.win_size=stft_args.win_size
        self.hop_size=stft_args.hop_size
        self.window=stft_args.window
        self.PEAQ_dir=PEAQ_dir
        self.PEMOQ_dir=PEMOQ_dir
        self.alg_dir=alg_dir




    def generate_inverse_window(self, stft_args):
        if stft_args.window=="hamming":
            return tf.signal.inverse_stft_window_fn(stft_args.hop_size, forward_window_fn=tf.signal.hamming_window)
        elif stft_args.window=="hann":
            return tf.signal.inverse_stft_window_fn(stft_args.hop_size, forward_window_fn=tf.signal.hann_window)
        elif stft_args.window=="kaiser_bessel":
            return tf.signal.inverse_stft_window_fn(stft_args.hop_size, forward_window_fn=tf.signal.kaiser_bessel_derived_window)
    def do_istft(self,data):
       
        window_fn = self.generate_inverse_window(self.stft_args)
        win_size=self.win_size
        hop_size=self.hop_size
        pred_cpx=data[...,0] + 1j * data[...,1]
        pred_time=tf.signal.inverse_stft(pred_cpx, win_size, hop_size, window_fn=window_fn)
        return pred_time

    def generate_images(self,cpx,name):
        spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
        spectrorgb=np.zeros(shape=(spectro.shape[0],spectro.shape[1],3))
        spectrorgb[...,0]=np.clip((np.flipud(np.transpose(10*np.log10(np.abs(cpx[...,0])+0.001)))+30)/50,0,1)
        spectrorgb[...,1]=np.clip((np.flipud(np.transpose(10*np.log10(np.abs(cpx[...,1])+0.001)))+30)/50,0,1)
        cmap=cv2.COLORMAP_JET
        spectro = np.array((1-spectro)* 255, dtype = np.uint8)
        spectro = cv2.applyColorMap(spectro, cmap)
        imageio.imwrite(os.path.join(self.test_results_filepath, name+".png"),spectro)
        spectrorgb = np.array(spectrorgb* 255, dtype = np.uint8)
        imageio.imwrite(os.path.join(self.test_results_filepath, name+"_ir.png"),spectrorgb)

    def generate_image_diff(self,clean , pred,name):
        difference=np.sqrt((clean[...,0]-pred[...,0])**2+(clean[...,1]-pred[...,1])**2)
        dif=np.clip(np.flipud(np.transpose(difference)),0,1)
        cmap=cv2.COLORMAP_JET
        dif = np.array((1-dif)* 255, dtype = np.uint8)
        dif = cv2.applyColorMap(dif, cmap)
        imageio.imwrite(os.path.join(self.test_results_filepath, name+"_diff.png"),dif)

    def inference_inner_classical(self, folder_name, method):
        nums=[]

        PEAQ_odg_noisy=[]
        PEAQ_odg_output=[]
        PEAQ_odg_diff=[]

        PEMOQ_odg_noisy=[]
        PEMOQ_odg_output=[]
        PEMOQ_odg_diff=[]

        SDR_noisy=[]
        SDR_output=[]
        SDR_diff=[]

        VGGish_noisy=[]
        VGGish_output=[]
        VGGish_diff=[]

        self.test_results_filepath = os.path.join(self.path_experiment,folder_name)
        if not os.path.exists(self.test_results_filepath):
            os.makedirs(self.test_results_filepath)
        num=0
        for element in tqdm(self.dataset_test.take(self.num_test_segments)):
            test_element=tf.data.Dataset.from_tensors(element)
            noisy_time=element[0].numpy()
            #noisy_time=self.do_istft(noisy)
            name_noisy=str(num)+'_noisy'
            clean_time=element[1].numpy()
            #clean_time=self.do_istft(clean)
            name_clean=str(num)+'_clean'
            print("inferencing")


            nums.append(num) 

            print("generating wavs")
            #noisy_time=noisy_time.numpy().astype(np.float32)
            noisy_time=noisy_time.astype(np.float32)
            wav_noisy_name_pre=os.path.join(self.test_results_filepath, name_noisy+"pre.wav")
            sf.write(wav_noisy_name_pre, noisy_time, 44100)

            #pred = self.model.predict(test_element.batch(1))
            name_pred=str(num)+'_output'
            wav_output_name_proc=os.path.join(self.test_results_filepath, name_pred+"proc.wav")
            self.process_in_matlab(wav_noisy_name_pre, wav_output_name_proc, method)

            noisy_time=noisy_time[44100::] #remove pre noise

            #clean_time=clean_time.numpy().astype(np.float32)
            clean_time=clean_time.astype(np.float32)
            clean_time=clean_time[44100::] #remove pre noise

            #change that !!!!
            #pred_time=self.do_istft(pred[0])
            #pred_time=pred_time.numpy().astype(np.float32)
            #pred_time=librosa.resample(np.transpose(pred_time),self.fs, 48000)
            #sf.write(wav_output_name, pred_time, 48000)
            #LOAD THE AUDIO!!!
            pred_time, sr=sf.read(wav_output_name_proc)
            assert sr==44100
            pred_time=pred_time[44100::] #remove prenoise
            
            #I am computing here the SDR at 48k, whle I was doing it before at 44.1k. I hope this won't cause any problem in the results. Consider resampling???
            SDR_t_noisy=10*np.log10(np.mean(np.square(clean_time))/np.mean(np.square(noisy_time-clean_time)))
            SDR_noisy.append(SDR_t_noisy)
            SDR_t_output=10*np.log10(np.mean(np.square(clean_time))/np.mean(np.square(pred_time-clean_time)))
            SDR_output.append(SDR_t_output)
            SDR_diff.append(SDR_t_output-SDR_t_noisy)
       
            noisy_time=librosa.resample(np.transpose(noisy_time),self.fs, 48000)     #P.Kabal PEAQ code is hardcoded at Fs=48000, so we have to resample
            wav_noisy_name=os.path.join(self.test_results_filepath, name_noisy+".wav")
            sf.write(wav_noisy_name, noisy_time, 48000) #overwrite without prenoise

            clean_time=librosa.resample(np.transpose(clean_time),self.fs, 48000)    #without prenoise please!!!
            wav_clean_name=os.path.join(self.test_results_filepath, name_clean+".wav")
            sf.write(wav_clean_name, clean_time, 48000)

            pred_time=librosa.resample(np.transpose(pred_time),self.fs, 48000)    #without prenoise please!!!
            wav_output_name=os.path.join(self.test_results_filepath, name_pred+".wav")
            sf.write(wav_output_name, pred_time, 48000)

            #save pred at 48k
            #print("calculating PEMOQ")
            #odg_noisy,odg_output =self.calculate_PEMOQ(wav_clean_name,wav_noisy_name,wav_output_name)
            #PEMOQ_odg_noisy.append(odg_noisy)
            #PEMOQ_odg_output.append(odg_output)
            #PEMOQ_odg_diff.append(odg_output-odg_noisy)

            #print("calculating PEAQ")
            #odg_noisy,odg_output =self.calculate_PEAQ(wav_clean_name,wav_noisy_name,wav_output_name)
            #PEAQ_odg_noisy.append(odg_noisy)
            #PEAQ_odg_output.append(odg_output)
            #PEAQ_odg_diff.append(odg_output-odg_noisy)
            
            print("calculating VGGish")
            VGGish_clean_embeddings=process_wav(wav_clean_name)
            VGGish_noisy_embeddings=process_wav(wav_noisy_name)
            VGGish_output_embeddings=process_wav(wav_output_name)
            dist_noisy = np.linalg.norm(VGGish_noisy_embeddings-VGGish_clean_embeddings)
            dist_output = np.linalg.norm(VGGish_output_embeddings-VGGish_clean_embeddings)
            VGGish_noisy.append(dist_noisy)
            VGGish_output.append(dist_output)
            VGGish_diff.append(-(dist_output-dist_noisy))
            os.remove(wav_clean_name)
            os.remove(wav_noisy_name)
            os.remove(wav_noisy_name_pre)
            os.remove(wav_output_name)
            os.remove(wav_output_name_proc)

            num=num+1

        frame = { 'num':nums,'PEAQ(ODG)_noisy': PEAQ_odg_noisy, 'PEAQ(ODG)_output': PEAQ_odg_output, 'PEAQ(ODG)_diff': PEAQ_odg_diff, 'PEMOQ(ODG)_noisy': PEMOQ_odg_noisy, 'PEMOQ(ODG)_output': PEMOQ_odg_output, 'PEMOQ(ODG)_diff': PEMOQ_odg_diff,'SDR_noisy': SDR_noisy, 'SDR_output': SDR_output, 'SDR_diff': SDR_diff, 'VGGish_noisy': VGGish_noisy, 'VGGish_output': VGGish_output,'VGGish_diff': VGGish_diff }

        metrics=pd.DataFrame(frame)    
        metrics.to_csv(os.path.join(self.test_results_filepath,"metrics.csv"),index=False)
        metrics=metrics.set_index('num')

        return metrics
    def inference_inner(self, folder_name):
        nums=[]

        PEAQ_odg_noisy=[]
        PEAQ_odg_output=[]
        PEAQ_odg_diff=[]

        PEMOQ_odg_noisy=[]
        PEMOQ_odg_output=[]
        PEMOQ_odg_diff=[]

        SDR_noisy=[]
        SDR_output=[]
        SDR_diff=[]

        VGGish_noisy=[]
        VGGish_output=[]
        VGGish_diff=[]

        self.test_results_filepath = os.path.join(self.path_experiment,folder_name)
        if not os.path.exists(self.test_results_filepath):
            os.makedirs(self.test_results_filepath)
        num=0
        for element in tqdm(self.dataset_test.take(self.num_test_segments)):
            test_element=tf.data.Dataset.from_tensors(element)
            noisy=element[0].numpy()
            noisy_time=self.do_istft(noisy)
            name_noisy=str(num)+'_noisy'
            clean=element[1].numpy()
            clean_time=self.do_istft(clean)
            name_clean=str(num)+'_clean'
            print("inferencing")
            pred = self.model.predict(test_element.batch(1))
            if self.args.unet.num_stages==2:
                pred=pred[0]
            pred_time=self.do_istft(pred[0])
            name_pred=str(num)+'_output'

            nums.append(num) 
            pred_time=pred_time.numpy().astype(np.float32)
            clean_time=clean_time.numpy().astype(np.float32)
            SDR_t_noisy=10*np.log10(np.mean(np.square(clean_time))/np.mean(np.square(noisy_time-clean_time)))
            SDR_t_output=10*np.log10(np.mean(np.square(clean_time))/np.mean(np.square(pred_time-clean_time)))
            SDR_noisy.append(SDR_t_noisy)
            SDR_output.append(SDR_t_output)
            SDR_diff.append(SDR_t_output-SDR_t_noisy)

            print("generating wavs")
            noisy_time=librosa.resample(np.transpose(noisy_time),self.fs, 48000)     #P.Kabal PEAQ code is hardcoded at Fs=48000, so we have to resample
            clean_time=librosa.resample(np.transpose(clean_time),self.fs, 48000)    
            pred_time=librosa.resample(np.transpose(pred_time),self.fs, 48000)

            wav_noisy_name=os.path.join(self.test_results_filepath, name_noisy+".wav")
            sf.write(wav_noisy_name, noisy_time, 48000)
            wav_clean_name=os.path.join(self.test_results_filepath, name_clean+".wav")
            sf.write(wav_clean_name, clean_time, 48000)
            wav_output_name=os.path.join(self.test_results_filepath, name_pred+".wav")
            sf.write(wav_output_name, pred_time, 48000)
       
            print("calculating PEMOQ")
            odg_noisy,odg_output =self.calculate_PEMOQ(wav_clean_name,wav_noisy_name,wav_output_name)
            PEMOQ_odg_noisy.append(odg_noisy)
            PEMOQ_odg_output.append(odg_output)
            PEMOQ_odg_diff.append(odg_output-odg_noisy)
            print("calculating PEAQ")
            odg_noisy,odg_output =self.calculate_PEAQ(wav_clean_name,wav_noisy_name,wav_output_name)
            PEAQ_odg_noisy.append(odg_noisy)
            PEAQ_odg_output.append(odg_output)
            PEAQ_odg_diff.append(odg_output-odg_noisy)
            
            print("calculating VGGish")
            VGGish_clean_embeddings=process_wav(wav_clean_name)
            VGGish_noisy_embeddings=process_wav(wav_noisy_name)
            VGGish_output_embeddings=process_wav(wav_output_name)
            dist_noisy = np.linalg.norm(VGGish_noisy_embeddings-VGGish_clean_embeddings)
            dist_output = np.linalg.norm(VGGish_output_embeddings-VGGish_clean_embeddings)
            VGGish_noisy.append(dist_noisy)
            VGGish_output.append(dist_output)
            VGGish_diff.append(-(dist_output-dist_noisy))
            os.remove(wav_clean_name)
            os.remove(wav_noisy_name)
            os.remove(wav_output_name)

            num=num+1

        frame = { 'num':nums,'PEAQ(ODG)_noisy': PEAQ_odg_noisy, 'PEAQ(ODG)_output': PEAQ_odg_output, 'PEAQ(ODG)_diff': PEAQ_odg_diff, 'PEMOQ(ODG)_noisy': PEMOQ_odg_noisy, 'PEMOQ(ODG)_output': PEMOQ_odg_output, 'PEMOQ(ODG)_diff': PEMOQ_odg_diff,'SDR_noisy': SDR_noisy, 'SDR_output': SDR_output, 'SDR_diff': SDR_diff, 'VGGish_noisy': VGGish_noisy, 'VGGish_output': VGGish_output,'VGGish_diff': VGGish_diff }

        metrics=pd.DataFrame(frame)    
        metrics.to_csv(os.path.join(self.test_results_filepath,"metrics.csv"),index=False)
        metrics=metrics.set_index('num')

        return metrics


    def inference_real(self, folder_name):
        self.test_results_filepath = os.path.join(self.path_experiment,folder_name)
        if not os.path.exists(self.test_results_filepath):
            os.makedirs(self.test_results_filepath)
        num=0
        for element in tqdm(self.dataset_real.take(self.num_real_test_segments)):
            test_element=tf.data.Dataset.from_tensors(element)
            noisy=element.numpy()
            noisy_time=self.do_istft(noisy)
            name_noisy="recording_"+str(num)+'_noisy.wav'
            pred = self.model.predict(test_element.batch(1))
            if self.args.unet.num_stages==2:
                pred=pred[0]
            pred_time=self.do_istft(pred[0])
            name_pred="recording_"+str(num)+'_output.wav'
            sf.write(os.path.join(self.test_results_filepath, name_noisy), noisy_time, self.fs)
            sf.write(os.path.join(self.test_results_filepath, name_pred), pred_time, self.fs)
            self.generate_images(noisy,name_noisy)
            self.generate_images(pred[0],name_pred)
            num=num+1


    def process_in_matlab(self,wav_noisy_name,wav_output_name,mode): #Opening and closing matlab to calculate PEAQ, rudimentary way to do it but easier. Make sure to have matlab installed
        addpath=self.alg_dir
        #odgmatfile_noisy=os.path.join(self.test_results_filepath, "odg_noisy.mat")
        #odgmatfile_pred=os.path.join(self.test_results_filepath, "odg_pred.mat")
        #bashCommand = "matlab -nodesktop -r 'addpath(\"PQevalAudio\", \"PQevalAudio/CB\",\"PQevalAudio/Misc\",\"PQevalAudio/MOV\", \"PQevalAudio/Patt\"), [odg, MOV]=PQevalAudio(\"0_clean_48.wav\",\"0_noise_48.wav\"), save(\"odg_noisy.mat\",\"odg\"), save(\"mov.mat\",\"MOV\") , exit'"
        bashCommand = "matlab -nodesktop -r 'addpath(genpath(\""+addpath+"\")), declick_and_denoise(\""+wav_noisy_name+"\",\""+wav_output_name+"\",\""+mode+"\") , exit'"
        print(bashCommand)
        p1 = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p1.communicate()  
    
        print(output)
        
        p1.wait()

    def calculate_PEMOQ(self,wav_clean_name,wav_noisy_name,wav_output_name): #Opening and closing matlab to calculate PEAQ, rudimentary way to do it but easier. Make sure to have matlab installed
        addpath=self.PEMOQ_dir
        odgmatfile_noisy=os.path.join(self.test_results_filepath, "odg_pemo_noisy.mat")
        odgmatfile_pred=os.path.join(self.test_results_filepath, "odg_pemo_pred.mat")
        #bashCommand = "matlab -nodesktop -r 'addpath(\"PQevalAudio\", \"PQevalAudio/CB\",\"PQevalAudio/Misc\",\"PQevalAudio/MOV\", \"PQevalAudio/Patt\"), [odg, MOV]=PQevalAudio(\"0_clean_48.wav\",\"0_noise_48.wav\"), save(\"odg_noisy.mat\",\"odg\"), save(\"mov.mat\",\"MOV\") , exit'"
        bashCommand = "matlab -nodesktop -r 'addpath(genpath(\""+addpath+"\")), [ ODG]=PEMOQ(\""+wav_clean_name+"\",\""+wav_noisy_name+"\"), save(\""+odgmatfile_noisy+"\",\"ODG\"), exit'"
        print(bashCommand)

        p1 = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p1.communicate()  
    
        print(output)
        
        bashCommand = "matlab -nodesktop -r 'addpath(genpath(\""+addpath+"\")), [ ODG]=PEMOQ(\""+wav_clean_name+"\",\""+wav_output_name+"\"), save(\""+odgmatfile_pred+"\",\"ODG\"), exit'"

        p2 = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p2.communicate()  
    
        print(output)
        p1.wait()
        p2.wait()
        #I save the odg results in a .mat file, which I load here. Not the most optimal method, sorry :/
        annots_noise = loadmat(odgmatfile_noisy)
        annots_pred = loadmat(odgmatfile_pred)
        #Consider loading also the movs!!
        return annots_noise["ODG"][0][0], annots_pred["ODG"][0][0]

    def calculate_PEAQ(self,wav_clean_name,wav_noisy_name,wav_output_name): #Opening and closing matlab to calculate PEAQ, rudimentary way to do it but easier. Make sure to have matlab installed
        addpath=self.PEAQ_dir
        odgmatfile_noisy=os.path.join(self.test_results_filepath, "odg_noisy.mat")
        odgmatfile_pred=os.path.join(self.test_results_filepath, "odg_pred.mat")
        #bashCommand = "matlab -nodesktop -r 'addpath(\"PQevalAudio\", \"PQevalAudio/CB\",\"PQevalAudio/Misc\",\"PQevalAudio/MOV\", \"PQevalAudio/Patt\"), [odg, MOV]=PQevalAudio(\"0_clean_48.wav\",\"0_noise_48.wav\"), save(\"odg_noisy.mat\",\"odg\"), save(\"mov.mat\",\"MOV\") , exit'"
        bashCommand = "matlab -nodesktop -r 'addpath(genpath(\""+addpath+"\")), [odg, MOV]=PQevalAudio(\""+wav_clean_name+"\",\""+wav_noisy_name+"\"), save(\""+odgmatfile_noisy+"\",\"odg\"), save(\"mov.mat\",\"MOV\") , exit'"
        p1 = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p1.communicate()  
    
        print(output)
        
        bashCommand = "matlab -nodesktop -r 'addpath(genpath(\""+addpath+"\")), [odg, MOV]=PQevalAudio(\""+wav_clean_name+"\",\""+wav_output_name+"\"), save(\""+odgmatfile_pred+"\",\"odg\"), save(\"mov.mat\",\"MOV\") , exit'"
        p2 = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        (output, err) = p2.communicate()  
    
        print(output)
        p1.wait()
        p2.wait()
        #I save the odg results in a .mat file, which I load here. Not the most optimal method, sorry :/
        annots_noise = loadmat(odgmatfile_noisy)
        annots_pred = loadmat(odgmatfile_pred)
        #Consider loading also the movs!!
        return annots_noise["odg"][0][0], annots_pred["odg"][0][0]

    def inference(self, name, method=None):
        print("Inferencing :",name)
        if self.dataset_test!=None:
            if method=="EM":    
                return self.inference_inner_classical(name, "EM")
            elif method=="wiener":
                return self.inference_inner_classical(name, "wiener")
            elif method=="wiener_declick":
                return self.inference_inner_classical(name, "wiener_declick")
            elif method=="EM_declick":
                return self.inference_inner_classical(name, "EM_declick")
            else:
                return self.inference_inner(name)
