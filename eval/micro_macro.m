function [micro, macro] = micro_macro(confusion_mat)
%MICRO_MACRO Summary of this function goes here
%   Detailed explanation goes here

    len=size(confusion_mat,1);
    macroTP=zeros(len,1);
    macroFP=zeros(len,1);
    macroFN=zeros(len,1);
    
    macroP=zeros(len,1);
    macroR=zeros(len,1);
    macroF=zeros(len,1);
    for i=1:len
        macroTP(i)=confusion_mat(i,i);
        macroFP(i)=sum(confusion_mat(:, i))-confusion_mat(i,i);
        macroFN(i)=sum(confusion_mat(i,:))-confusion_mat(i,i);
        macroP(i)=macroTP(i)/(macroTP(i)+macroFP(i));
        macroR(i)=macroTP(i)/(macroTP(i)+macroFN(i));
        macroF(i)=2*macroP(i)*macroR(i)/(macroP(i)+macroR(i));
    end
    
    macro.precision=mean(macroP);
    macro.recall=mean(macroR);
    macro.fscore=mean(macroF);

    
    micro.precision=sum(macroTP)/(sum(macroTP)+sum(macroFP));
    micro.recall=sum(macroTP)/(sum(macroTP)+sum(macroFN));
    micro.fscore=2*micro.precision*micro.recall/(micro.precision+micro.recall);

end

