# A two-stage U-Net for high-fidelity denoising of historical recordings

Official repository of the paper:

> E. Moliner and V. Välimäki,, "A two-stage U-Net for high-fidelity denosing of historical recordinds", in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Singapore, May, 2022

## Abstract
Enhancing the sound quality of historical music recordings is a long-standing problem. This paper presents a novel denoising method based on a fully-convolutional deep neural network. A two-stage U-Net model architecture is designed to model and suppress the degradations with high fidelity. The method processes the time-frequency representation of audio, and is trained using realistic noisy data to jointly remove hiss, clicks, thumps, and other common additive disturbances from old analog discs. The proposed model outperforms previous methods in both objective and subjective metrics. The results of a formal blind listening test show that the method can denoise real gramophone recordings with an excellent quality. This study shows the importance of realistic training data and the power of deep learning in audio restoration.

<p align="center">
<img src="https://user-images.githubusercontent.com/64018465/131505025-e4530f55-fe5d-4bf4-ae64-cc9a502e5874.png" alt="Schema represention"
width="400px"></p>

Listen to our audio samples [link to url]
## Requirements
You will need at least python 3.7 and CUDA 10.1 if you want to use GPU. See `requirements.txt` for the required package versions.

To install the environment through anaconda, follow the instructions:

    conda env update -f environment.yml
    conda activate historical_denoiser

## Denosing Recordings
Run the following commands to install the pretrained weights of the two-stage U-Net model:

    wget ckptzipurl 
    unzip name.zip /experiments/trained_model
    
If the environment is installed correctly, you can denoise an audio file by running:

    bash inference.sh $filename
    
A ".wav" file with the denoised version, as well as the residual noise and the original signal in "mono" will be stored in the same directory as the input file.
## Training
