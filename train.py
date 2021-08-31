import os
import hydra
import logging

logger = logging.getLogger(__name__)

def run(args):
    import unet
    import tensorflow as tf
    import  dataset_loader
    from tensorflow.keras.optimizers import Adam
    import soundfile as sf
    import datetime
    from tqdm import tqdm
    import numpy as np

    path_experiment=str(args.path_experiment)

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    
    path_music_train=args.dset.path_music_train
    path_music_test=args.dset.path_music_test
    path_music_validation=args.dset.path_music_validation

    path_noise=args.dset.path_noise
    path_recordings=args.dset.path_recordings
    
    fs=args.fs
    overlap=args.overlap
    seg_len_s_train=args.seg_len_s_train

    batch_size=args.batch_size
    epochs=args.epochs

    num_real_test_segments=args.num_real_test_segments
    buffer_size=args.buffer_size #for shuffle
    
    tensorboard_logs=args.tensorboard_logs
    
    def do_stft(noisy, clean=None):
        
        window_fn = tf.signal.hamming_window

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
    
    #Loading data. The train dataset object is a generator. The validation dataset is loaded in memory.

    dataset_train, dataset_val=dataset_loader.load_data(buffer_size, path_music_train, path_music_validation, path_noise, fs=fs, seg_len_s=seg_len_s_train)

    dataset_train=dataset_train.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)
    dataset_val=dataset_val.map(do_stft, num_parallel_calls=args.num_workers, deterministic=None)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        #build the model
        unet_model = unet.build_model_denoise(unet_args=args.unet)

        current_lr=args.lr
        optimizer = Adam(learning_rate=current_lr, beta_1=args.beta1, beta_2=args.beta2)
        
        loss=tf.keras.losses.MeanAbsoluteError()

    if args.use_tensorboard:
        log_dir = os.path.join(tensorboard_logs, os.path.basename(path_experiment)+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_summary_writer = tf.summary.create_file_writer(log_dir+"/train")
        val_summary_writer = tf.summary.create_file_writer(log_dir+"/validation")
    
    #path where the checkpoints will be saved
    checkpoint_filepath=os.path.join(path_experiment, 'checkpoint')
    
    dataset_train=dataset_train.batch(batch_size)
    dataset_val=dataset_val.batch(batch_size)

    #prefetching the dataset for better performance
    dataset_train=dataset_train.prefetch(batch_size*20)
    dataset_val=dataset_val.prefetch(batch_size*20)

    dataset_train=strategy.experimental_distribute_dataset(dataset_train)
    dataset_val=strategy.experimental_distribute_dataset(dataset_val)

    iterator = iter(dataset_train)

    from trainer import Trainer

    trainer=Trainer(unet_model,optimizer,loss,strategy, path_experiment,  args)

    for epoch in range(epochs):
        total_loss=0
        step_loss=0
        for step in tqdm(range(args.steps_per_epoch), desc="Training epoch "+str(epoch)):
            step_loss=trainer.distributed_training_step(iterator.get_next())
            total_loss+=step_loss
            with train_summary_writer.as_default():
                tf.summary.scalar('batch_loss', step_loss, step=step)
                tf.summary.scalar('batch_mean_absolute_error', trainer.train_mae.result(), step=step)

        train_loss=total_loss/args.steps_per_epoch       

        for x in tqdm(dataset_val, desc="Validating epoch "+str(epoch)):
            trainer.distributed_test_step(x)

        template = ("Epoch {}, Loss: {}, train_MAE: {}, val_Loss: {}, val_MAE: {}")
        print (template.format(epoch+1, train_loss, trainer.train_mae.result(), trainer.val_loss.result(), trainer.val_mae.result()))

        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', train_loss, step=epoch)
            tf.summary.scalar('epoch_mean_absolute_error', trainer.train_mae.result(), step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', trainer.val_loss.result(), step=epoch)
            tf.summary.scalar('epoch_mean_absolute_error', trainer.val_mae.result(), step=epoch)

        trainer.train_mae.reset_states()
        trainer.val_loss.reset_states()
        trainer.val_mae.reset_states()
         
        if (epoch+1) % 50 == 0:
            if args.variable_lr:
                current_lr*=1e-1
                trainer.optimizer.lr=current_lr
            try: 
                unet_model.save_weights(checkpoint_filpath)
            except:
                pass

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







