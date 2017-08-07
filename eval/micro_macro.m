%    Copyright (C) 2017  Joseph St.Amand

%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.

%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.

%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

