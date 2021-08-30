clear; clc; 
%addpath('PQevalAudio', 'PQevalAudio/CB','PQevalAudio/Misc','PQevalAudio/MOV', 'PQevalAudio/Patt')  

ref = '2_clean_48.wav' % 16 times oversampling
test = '2_noise_48.wav' % 4 times oversampling
%test = '../2_output_48.wav' % 4 times oversampling


[odg, movb] = PQevalAudio(ref, test)
