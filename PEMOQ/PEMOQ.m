function [ODG]=PEMOQ(ref,test)
[refa, fs]=audioread(ref);
[testa, fs]=audioread(test);
[PSM, PSMt, ODG, PSM_inst] = audioqual(refa, testa, fs);

