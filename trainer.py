
import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from tqdm import tqdm
import pandas as pd                                    

class Trainer():
    def __init__(self, model,  optimizer,loss,  strategy, path_experiment,  args):
        self.model=model
        print(self.model.summary())
        self.strategy=strategy
        self.optimizer=optimizer
        self.path_experiment=path_experiment
        self.args=args
        #self.metrics=[]

        with self.strategy.scope():
            #loss_fn=tf.keras.losses.mean_absolute_error
            loss.reduction=tf.keras.losses.Reduction.NONE
            self.loss_object=loss
            self.train_mae_s1=tf.keras.metrics.MeanAbsoluteError(name="train_mae_s1")
            self.train_mae=tf.keras.metrics.MeanAbsoluteError(name="train_mae_s2")
            self.val_mae=tf.keras.metrics.MeanAbsoluteError(name="validation_mae")
            self.val_loss = tf.keras.metrics.Mean(name='test_loss')


    def train_step(self,inputs):
        noisy, clean= inputs
    
        with tf.GradientTape() as tape:
    
            logits_2,logits_1 = self.model(noisy, training=True)  # Logits for this minibatch
    
            loss_value = tf.reduce_mean(self.loss_object(clean, logits_2) + tf.reduce_mean(self.loss_object(clean, logits_1)))
    
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_mae.update_state(clean, logits_2)
        self.train_mae_s1.update_state(clean, logits_1)
        return loss_value

    def test_step(self,inputs):

        noisy,clean = inputs
     
        predictions_s2, predictions_s1 = self.model(noisy, training=False)
        t_loss = self.loss_object(clean, predictions_s2)+self.loss_object(clean, predictions_s1)
     
        self.val_mae.update_state(clean,predictions_s2)
        self.val_loss.update_state(t_loss)

    @tf.function()
    def distributed_training_step(self,inputs):
        per_replica_losses=self.strategy.run(self.train_step, args=(inputs,))
        reduced_losses=self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return reduced_losses

    @tf.function
    def distributed_test_step(self,inputs):
        return self.strategy.run(self.test_step, args=(inputs,))


        

