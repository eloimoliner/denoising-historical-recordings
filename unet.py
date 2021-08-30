import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal
import math as m

def build_model_denoise(unet_args=None):

    inputs=Input(shape=(None, None,2))

    outputs_stage_2,outputs_stage_1=MultiStage_denoise(unet_args=unet_args)(inputs)

    #Encapsulating MultiStage_denoise in a keras.Model object
    model= tf.keras.Model(inputs=inputs,outputs=[outputs_stage_2, outputs_stage_1])

    return model
class DenseBlock(layers.Layer):
    '''
    [B, T, F, N] => [B, T, F, N] 
    DenseNet Block consisting of "num_layers" densely connected convolutional layers
    '''
    def __init__(self, num_layers, N, ksize,activation):
        '''
        num_layers:     number of densely connected conv. layers
        N:              Number of filters (same in each layer) 
        ksize:          Kernel size (same in each layer) 
        '''
        super(DenseBlock, self).__init__()
        self.activation=activation

        self.paddings_1=get_paddings(ksize)
        self.H=[]
        self.num_layers=num_layers

        for i in range(num_layers):
            self.H.append(layers.Conv2D(filters=N,
                                      kernel_size=ksize,
                                      kernel_initializer=TruncatedNormal(),
                                      strides=1,
                                      padding='VALID',
                                      activation=self.activation))

    def call(self, x):

        x_=tf.pad(x, self.paddings_1, mode='SYMMETRIC')
        x_ = self.H[0](x_)
        if self.num_layers>1:
            for h in self.H[1:]:
                x = tf.concat([x_, x], axis=-1)
                x_=tf.pad(x, self.paddings_1, mode='SYMMETRIC')
                x_ = h(x_)  

        return x_


class FinalBlock(layers.Layer):
    '''
    [B, T, F, N] => [B, T, F, 2] 
    Final block. Basically, a 3x3 conv. layer to map the output features to the output complex spectrogram.

    '''
    def __init__(self):
        super(FinalBlock, self).__init__()
        ksize=(3,3)
        self.paddings_2=get_paddings(ksize)
        self.conv2=layers.Conv2D(filters=2,
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=None)


    def call(self, inputs ):

        x=tf.pad(inputs, self.paddings_2, mode='SYMMETRIC')
        pred=self.conv2(x)

        return pred
class SAM(layers.Layer):
    '''
    [B, T, F, N] => [B, T, F, N] , [B, T, F, N]
    Supervised Attention Module:
    The purpose of SAM is to make the network only propagate the most relevant features to the second stage, discarding the less useful ones.
    The estimated residual noise signal is generated from the U-Net output features by means of a 3x3 convolutional layer. 
    The first stage output is then calculated adding the original input spectrogram to the residual noise. 
    The attention-guided features are computed using the attention masks M, which are directly calculated from the first stage output with a 1x1 convolution and a sigmoid function. 

    '''
    def __init__(self, n_feat):
        super(SAM, self).__init__()

        ksize=(3,3)
        self.paddings_1=get_paddings(ksize)
        self.conv1 = layers.Conv2D(filters=n_feat,
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=None)
        ksize=(3,3)
        self.paddings_2=get_paddings(ksize)
        self.conv2=layers.Conv2D(filters=2,
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=None)

        ksize=(3,3)
        self.paddings_3=get_paddings(ksize)
        self.conv3 = layers.Conv2D(filters=n_feat,
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=None)
        self.cropadd=CropAddBlock()

    def call(self, inputs, input_spectrogram):
        x1=tf.pad(inputs, self.paddings_1, mode='SYMMETRIC')
        x1 = self.conv1(x1)

        x=tf.pad(inputs, self.paddings_2, mode='SYMMETRIC')
        x=self.conv2(x)

        #residual prediction
        pred = layers.Add()([x, input_spectrogram]) #features to next stage

        x3=tf.pad(pred, self.paddings_3, mode='SYMMETRIC')
        M=self.conv3(x3)

        M= tf.keras.activations.sigmoid(M)
        x1=layers.Multiply()([x1, M])
        x1 = layers.Add()([x1, inputs]) #features to next stage

        return x1, pred


class AddFreqEncoding(layers.Layer):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim):
        super(AddFreqEncoding, self).__init__()
        pi = tf.constant(m.pi)
        pi=tf.cast(pi,'float32')
        self.f_dim=f_dim #f_dim is fixed
        n=tf.cast(tf.range(f_dim)/(f_dim-1),'float32')
        coss=tf.math.cos(pi*n)
        f_channel = tf.expand_dims(coss, -1) #(1025,1)
        self.fembeddings= f_channel
        
        for k in range(1,10):   
            coss=tf.math.cos(2**k*pi*n)
            f_channel = tf.expand_dims(coss, -1) #(1025,1)
            self.fembeddings=tf.concat([self.fembeddings,f_channel],axis=-1) #(1025,10)
    

    def call(self, input_tensor):

        batch_size_tensor = tf.shape(input_tensor)[0]  # get batch size
        time_dim = tf.shape(input_tensor)[1]  # get time dimension

        fembeddings_2 = tf.broadcast_to(self.fembeddings, [batch_size_tensor, time_dim, self.f_dim, 10])
    
        
        return tf.concat([input_tensor,fembeddings_2],axis=-1)  #(batch,427,1025,12)


def get_paddings(K):
        return tf.constant([[0,0],[K[0]//2, K[0]//2 -(1- K[0]%2) ], [  K[1]//2, K[1]//2 -(1- K[1]%2)  ],[0,0]])

class Decoder(layers.Layer):
    '''
    [B, T, F, N] , skip connections => [B, T, F, N]  
    Decoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, unet_args):
        super(Decoder, self).__init__()

        self.Ns=Ns
        self.Ss=Ss
        self.activation=unet_args.activation
        self.depth=unet_args.depth


        ksize=(3,3)
        self.paddings_3=get_paddings(ksize)
        self.conv2d_3=layers.Conv2D(filters=self.Ns[self.depth],
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=self.activation)

        self.cropadd=CropAddBlock()

        self.dblocks=[]
        for i in range(self.depth):
            self.dblocks.append(D_Block(layer_idx=i,N=self.Ns[i], S=self.Ss[i], activation=self.activation,num_tfc=unet_args.num_tfc))

    def call(self,inputs, contracting_layers):
        x=inputs
        for i in range(self.depth,0,-1):
            x=self.dblocks[i-1](x, contracting_layers[i-1])
        return x 

class Encoder(tf.keras.Model):

    '''
    [B, T, F, N] => skip connections , [B, T, F, N_4]  
    Encoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, unet_args):
        super(Encoder, self).__init__()
        self.Ns=Ns
        self.Ss=Ss
        self.activation=unet_args.activation
        self.depth=unet_args.depth

        self.contracting_layers = {}

        self.eblocks=[]
        for i in range(self.depth):
            self.eblocks.append(E_Block(layer_idx=i,N0=self.Ns[i],N=self.Ns[i+1],S=self.Ss[i], activation=self.activation , num_tfc=unet_args.num_tfc))

        self.i_block=I_Block(self.Ns[self.depth],self.activation,unet_args.num_tfc)

    def call(self, inputs):
        x=inputs
        for i in range(self.depth):

            x, x_contract=self.eblocks[i](x)
        
            self.contracting_layers[i] = x_contract #if remove 0, correct this
        x=self.i_block(x)

        return x, self.contracting_layers

class MultiStage_denoise(tf.keras.Model):

    def __init__(self,  unet_args=None):
        super(MultiStage_denoise, self).__init__()

        self.activation=unet_args.activation
        self.depth=unet_args.depth
        if unet_args.use_fencoding:
            self.freq_encoding=AddFreqEncoding(unet_args.f_dim)
        self.use_sam=unet_args.use_SAM
        self.use_fencoding=unet_args.use_fencoding
        self.num_stages=unet_args.num_stages
        #Encoder
        self.Ns= [32,64,64,128,128,256,512] 
        self.Ss= [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)]
        
        #initial feature extractor
        ksize=(7,7)
        self.paddings_1=get_paddings(ksize)
        self.conv2d_1 = layers.Conv2D(filters=self.Ns[0],
                      kernel_size=ksize,
                      kernel_initializer=TruncatedNormal(),
                      strides=1, 
                      padding='VALID',
                      activation=self.activation)


        self.encoder_s1=Encoder(self.Ns, self.Ss, unet_args)
        self.decoder_s1=Decoder(self.Ns, self.Ss, unet_args)

        self.cropconcat = CropConcatBlock()
        self.cropadd = CropAddBlock()

        self.finalblock=FinalBlock()

        if self.num_stages>1:
            self.sam_1=SAM(self.Ns[0])

            #initial feature extractor
            ksize=(7,7)
            self.paddings_2=get_paddings(ksize)
            self.conv2d_2 = layers.Conv2D(filters=self.Ns[0],
                          kernel_size=ksize,
                          kernel_initializer=TruncatedNormal(),
                          strides=1, 
                          padding='VALID',
                          activation=self.activation)
            

            self.encoder_s2=Encoder(self.Ns, self.Ss, unet_args)
            self.decoder_s2=Decoder(self.Ns, self.Ss, unet_args)

    @tf.function()
    def call(self, inputs):
        
        if self.use_fencoding:
            x_w_freq=self.freq_encoding(inputs)   #None, None, 1025, 12 
        else:
            x_w_freq=inputs

        #intitial feature extractor
        x=tf.pad(x_w_freq, self.paddings_1, mode='SYMMETRIC')
        x=self.conv2d_1(x) #None, None, 1025, 32

        x, contracting_layers_s1= self.encoder_s1(x)
        #decoder
        feats_s1 =self.decoder_s1(x, contracting_layers_s1) #None, None, 1025, 32 features

        if self.num_stages>1:        
            #SAM module
            Fout, pred_stage_1=self.sam_1(feats_s1,inputs)
                
            #intitial feature extractor
            x=tf.pad(x_w_freq, self.paddings_2, mode='SYMMETRIC')
            x=self.conv2d_2(x)
    
            if self.use_sam:
                x = tf.concat([x, Fout], axis=-1)
            else:
                x = tf.concat([x,feats_s1], axis=-1)
    
            x, contracting_layers_s2= self.encoder_s2(x)

            feats_s2=self.decoder_s2(x, contracting_layers_s2) #None, None, 1025, 32 features
            
            #consider implementing a third stage?

            pred_stage_2=self.finalblock(feats_s2) 
            return pred_stage_2, pred_stage_1
        else:             
            pred_stage_1=self.finalblock(feats_s1) 
            return pred_stage_1
            
class I_Block(layers.Layer):
    '''
    [B, T, F, N] => [B, T, F, N] 
    Intermediate block:
    Basically, a densenet block with a residual connection
    '''
    def __init__(self,N,activation, num_tfc, **kwargs):
        super(I_Block, self).__init__(**kwargs)

        ksize=(3,3)
        self.tfc=DenseBlock(num_tfc,N,ksize, activation)

        self.conv2d_res= layers.Conv2D(filters=N,
                                      kernel_size=(1,1),
                                      kernel_initializer=TruncatedNormal(),
                                      strides=1,
                                      padding='VALID')

    def call(self,inputs):
        x=self.tfc(inputs)

        inputs_proj=self.conv2d_res(inputs)
        return layers.Add()([x,inputs_proj])


class E_Block(layers.Layer):

    def __init__(self, layer_idx,N0, N,  S,activation, num_tfc, **kwargs):
        super(E_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N0=N0
        self.N=N
        self.S=S
        self.activation=activation
        self.i_block=I_Block(N0,activation,num_tfc)

        ksize=(S[0]+2,S[1]+2)
        self.paddings_2=get_paddings(ksize)
        self.conv2d_2 = layers.Conv2D(filters=N,
                                      kernel_size=(S[0]+2,S[1]+2),
                                      kernel_initializer=TruncatedNormal(),
                                      strides=S,
                                      padding='VALID',
                                      activation=self.activation)


    def call(self, inputs, training=None, **kwargs):
        x=self.i_block(inputs)
        
        x_down=tf.pad(x, self.paddings_2, mode='SYMMETRIC')
        x_down = self.conv2d_2(x_down)

        return x_down, x


    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    N=self.N,
                    S=self.S,
                    **super(E_Block, self).get_config()
                    )
class D_Block(layers.Layer):

    def __init__(self, layer_idx, N,  S,activation,  num_tfc, **kwargs):
        super(D_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N=N
        self.S=S
        self.activation=activation
        ksize=(S[0]+2, S[1]+2)
        self.paddings_1=get_paddings(ksize)

        self.tconv_1= layers.Conv2DTranspose(filters=N,
                                             kernel_size=(S[0]+2, S[1]+2),
                                             kernel_initializer=TruncatedNormal(),
                                             strides=S,
                                             activation=self.activation,
                                             padding='VALID')

        self.upsampling = layers.UpSampling2D(size=S,  interpolation='nearest')

        self.projection = layers.Conv2D(filters=N,
                                      kernel_size=(1,1),
                                      kernel_initializer=TruncatedNormal(),
                                      strides=1,
                                      activation=self.activation,
                                      padding='VALID')
        self.cropadd=CropAddBlock()
        self.cropconcat=CropConcatBlock()

        self.i_block=I_Block(N,activation,num_tfc)

    def call(self, inputs, bridge, previous_encoder=None, previous_decoder=None,**kwargs):
        x = inputs
        x=tf.pad(x, self.paddings_1, mode='SYMMETRIC')
        x = self.tconv_1(inputs)

        x2= self.upsampling(inputs)

        if x2.shape[-1]!=x.shape[-1]:
            x2= self.projection(x2)

        x= self.cropadd(x,x2)


        x=self.cropconcat(x,bridge)

        x=self.i_block(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    N=self.N,
                    S=self.S,
                    **super(D_Block, self).get_config()
                    )

class CropAddBlock(layers.Layer):

    def call(self,down_layer, x,  **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)


        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = layers.Add()([down_layer_cropped, x])
        return x

class CropConcatBlock(layers.Layer):

    def call(self, down_layer, x, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x2_shape[1] + height_diff),
                                        width_diff: (x2_shape[2] + width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x
