import os
import hydra
import logging
'''
Script used for the objective experiments
WARNING: it calls MATLAB to calculate PEAQ and PEMO-Q. The whole process may be very slow
'''
logger = logging.getLogger(__name__)

def run(args):
    import unet
    import dataset_loader
    import tensorflow as tf
    import pandas as pd
    
    path_experiment=str(args.path_experiment)

    print(path_experiment)
    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    unet_model = unet.build_model_denoise(stereo=stereo,unet_args=args.unet)

    ckpt=os.path.join(path_experiment, 'checkpoint')
    unet_model.load_weights(ckpt)


    path_pianos_test=args.dset.path_piano_test
    path_strings_test=args.dset.path_strings_test
    path_orchestra_test=args.dset.path_orchestra_test
    path_opera_test=args.dset.path_opera_test
    path_noise=args.dset.path_noise
    fs=args.fs
    seg_len_s=20
    numsamples=1000//seg_len_s
      
    def do_stft(noisy, clean=None):
        
        if args.stft.window=="hamming":
            window_fn = tf.signal.hamming_window
        elif args.stft.window=="hann":
            window_fn=tf.signal.hann_window
        elif args.stft.window=="kaiser_bessel":
            window_fn=tf.signal.kaiser_bessel_derived_window

        win_size=args.stft.win_size
        hop_size=args.stft.hop_size

        
        stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
        stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
        
        if clean!=None:

            stft_signal_clean=tf.signal.stft(clean,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
            stft_clean_stacked=tf.stack( values=[tf.math.real(stft_signal_clean), tf.math.imag(stft_signal_clean)], axis=-1)


            return stft_noisy_stacked, stft_clean_stacked
        else:
    
            return stft_noisy_stacked

    from tester import Tester

    testPath=os.path.join(path_experiment,"final_test")
    if not os.path.exists(testPath):
        os.makedirs(testPath)

    tester=Tester(unet_model, testPath,  args)

    PEAQ_dir="/scratch/work/molinee2/unet_dir/unet_historical_music/PQevalAudio"
    PEMOQ_dir="/scratch/work/molinee2/unet_dir/unet_historical_music/PEMOQ"

    dataset_test_pianos=dataset_loader.load_data_formal( path_pianos_test, path_noise, noise_amount="mid_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_pianos=dataset_test_pianos.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_pianos,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("pianos_midsnr")

    dataset_test_strings=dataset_loader.load_data_formal( path_strings_test, path_noise,noise_amount="mid_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_strings=dataset_test_strings.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_strings,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("strings_midsnr")

    dataset_test_orchestra=dataset_loader.load_data_formal( path_orchestra_test, path_noise, noise_amount="mid_snr", num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_orchestra=dataset_test_orchestra.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_orchestra,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("orchestra_midsnr")

    dataset_test_opera=dataset_loader.load_data_formal( path_opera_test, path_noise, noise_amount="mid_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_opera=dataset_test_opera.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_opera,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("opera_midsnr")

    dataset_test_strings=dataset_loader.load_data_formal( path_strings_test, path_noise,noise_amount="low_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_strings=dataset_test_strings.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_strings,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("strings_lowsnr")

    dataset_test_orchestra=dataset_loader.load_data_formal( path_orchestra_test, path_noise,noise_amount="low_snr", num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_orchestra=dataset_test_orchestra.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_orchestra,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("orchestra_lowsnr")

    dataset_test_opera=dataset_loader.load_data_formal( path_opera_test, path_noise, noise_amount="low_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_opera=dataset_test_opera.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_opera,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("opera_lowsnr")


    dataset_test_pianos=dataset_loader.load_data_formal( path_pianos_test, path_noise, noise_amount="low_snr",num_samples=numsamples, fs=fs, seg_len_s=seg_len_s, stereo=stereo)
    dataset_test_pianos=dataset_test_pianos.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    tester.init_inference(dataset_test_pianos,numsamples,fs,args.stft, PEAQ_dir, PEMOQ_dir=PEMOQ_dir)
    metrics=tester.inference("pianos_lowsnr")
    

    names=["strings_midsnr","strings_lowsnr","opera_midsnr","opera_lowsnr","pianos_midsnr","pianos_lowsnr","orchestra_midsnr","orchestra_lowsnr"]
    for n in names:
        a=pd.read_csv(os.path.join(testPath,n,"metrics.csv"))
        meanPEAQ=a["PEAQ(ODG)_diff"].sum()/50
        meanPEMOQ=a["PEMOQ(ODG)_diff"].sum()/50
        meanSDR=a["SDR_diff"].sum()/50
        print(n,": PEAQ ",str(meanPEAQ), "PEMOQ ", str(meanPEMOQ), "SDR ", str(meanSDR))

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







