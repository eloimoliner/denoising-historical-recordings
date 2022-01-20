import os
import hydra
import logging

logger = logging.getLogger(__name__)

def run(args):
    import unet
    import tensorflow as tf
    import soundfile as sf
    import numpy as np
    from tqdm import tqdm
    import scipy.signal
    
    path_experiment=str(args.path_experiment)

    unet_model = unet.build_model_denoise(unet_args=args.unet)

    ckpt=os.path.join(os.path.dirname(os.path.abspath(__file__)),path_experiment, 'checkpoint')
    unet_model.load_weights(ckpt)

    def do_stft(noisy):
        
        window_fn = tf.signal.hamming_window

        win_size=args.stft.win_size
        hop_size=args.stft.hop_size

        
        stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size, pad_end=True)
        stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    
        return stft_noisy_stacked

    def do_istft(data):
        
        window_fn = tf.signal.hamming_window

        win_size=args.stft.win_size
        hop_size=args.stft.hop_size

        inv_window_fn=tf.signal.inverse_stft_window_fn(hop_size, forward_window_fn=window_fn)

        pred_cpx=data[...,0] + 1j * data[...,1]
        pred_time=tf.signal.inverse_stft(pred_cpx, win_size, hop_size, window_fn=inv_window_fn)
        return pred_time

    audio=str(args.inference.audio)
    data, samplerate = sf.read(audio)
    print(data.dtype)
    #Stereo to mono
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=44100: 
        print("Resampling")
   
        data=scipy.signal.resample(data, int((44100  / samplerate )*len(data))+1)  
 
    
    
    segment_size=44100*5  #20s segments

    length_data=len(data)
    overlapsize=2048 #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    audio_finished=False
    pointer=0
    denoised_data=np.zeros(shape=(len(data),))
    residual_noise=np.zeros(shape=(len(data),))
    numchunks=int(np.ceil(length_data/segment_size))
     
    for i in tqdm(range(numchunks)):
        if pointer+segment_size<length_data:
            segment=data[pointer:pointer+segment_size]
            #dostft
            segment_TF=do_stft(segment)
            segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)
            pred = unet_model.predict(segment_TF_ds.batch(1))
            pred=pred[0]
            residual=segment_TF-pred[0]
            residual=np.array(residual)
            pred_time=do_istft(pred[0])
            residual_time=do_istft(residual)
            residual_time=np.array(residual_time)

            if pointer==0:
                pred_time=np.concatenate((pred_time[0:int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
                residual_time=np.concatenate((residual_time[0:int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(pred_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(residual_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                
            denoised_data[pointer:pointer+segment_size]=denoised_data[pointer:pointer+segment_size]+pred_time
            residual_noise[pointer:pointer+segment_size]=residual_noise[pointer:pointer+segment_size]+residual_time

            pointer=pointer+segment_size-overlapsize
        else: 
            segment=data[pointer::]
            lensegment=len(segment)
            segment=np.concatenate((segment, np.zeros(shape=(int(segment_size-len(segment)),))), axis=0)
            audio_finished=True
            #dostft
            segment_TF=do_stft(segment)

            segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)

            pred = unet_model.predict(segment_TF_ds.batch(1))
            pred=pred[0]
            residual=segment_TF-pred[0]
            residual=np.array(residual)
            pred_time=do_istft(pred[0])
            pred_time=np.array(pred_time)
            pred_time=pred_time[0:segment_size]
            residual_time=do_istft(residual)
            residual_time=np.array(residual_time)
            residual_time=residual_time[0:segment_size]
            if pointer==0:
                pred_time=pred_time
                residual_time=residual_time
            else:
                pred_time=np.concatenate((np.multiply(pred_time[0:int(overlapsize)], window_left), pred_time[int(overlapsize):int(segment_size)]),axis=0)
                residual_time=np.concatenate((np.multiply(residual_time[0:int(overlapsize)], window_left), residual_time[int(overlapsize):int(segment_size)]),axis=0)

            denoised_data[pointer::]=denoised_data[pointer::]+pred_time[0:lensegment]
            residual_noise[pointer::]=residual_noise[pointer::]+residual_time[0:lensegment]

    basename=os.path.splitext(audio)[0]
    wav_noisy_name=basename+"_noisy_input"+".wav"
    sf.write(wav_noisy_name, data, 44100)
    wav_output_name=basename+"_denoised"+".wav"
    sf.write(wav_output_name, denoised_data, 44100)
    wav_output_name=basename+"_residual"+".wav"
    sf.write(wav_output_name, residual_noise, 44100)
    

def _main(args):
    global __file__

    __file__ = hydra.utils.to_absolute_path(__file__)

    run(args)


@hydra.main(config_path="conf/conf.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()







