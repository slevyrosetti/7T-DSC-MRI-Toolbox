function [BAT, TT, TTP, rTTP, PR] = dsc_map_BAT_TT_TTP_rTTP_PR(conc,mask,options,injectionTime,TE,r2GdInBlood)
%Homemade function
%Author: Simon Levy
%
%Computes the MTT voxel-wise (duration of positive concentration values)
%and Time-To-Peak (TTP) (time between injection and bolus peak).
%
%Inputs: conc (Matrice 4D) che contiene gli andamenti delle
%concentrazioni DSC di tutti i voxel.
%Options � la sruct che contiene le opzioni del metodo, quelli
%significativi sono:
%
%options.time - Rappresenta il vettore dei tempi dell'esame DSC (ogni
%               campione rappresenta l'acquisizione di un intero volume cerebrale
%
%options.par.kh - Parametro che rappresenta la dipendenza dall'ematocrito
%                 del Cerebral Blood Volume (CBV), per default � settato ad
%                 uno, in questo caso si ottengono stime RELATIVE del
%                 parametro CBV.
%
%options.par.rho - Parametro che rappresenta la dipendenza dalla densit�
%                  del sangue del Cerebral Blood Volume (CBV), per default
%                  � settato ad uno, in questo caso si ottengono stime
%                  RELATIVE del parametro CBV.
%
%Parametri in uscita:
%cbv - (matrice 3D) che contiene al suo interno la mappa parametrica
%      calcolata

% defaults
if ~exist('TE','var'), TE = 42; end
if ~exist('r2GdInBlood','var'), r2GdInBlood = 3.55; end
 
if options.display > 0
    fprintf('\n   rTT (duration of positive concentration curve)\n');
    fprintf('\n   TTP (time between injection and bolus peak)\n');
end

BAT=zeros(options.nR,options.nC,options.nS);
TT=zeros(options.nR,options.nC,options.nS);
TTP=zeros(options.nR,options.nC,options.nS);
rTTP=zeros(options.nR,options.nC,options.nS);

if options.waitbar
    hw_rMTT=waitbar(0,'Calculating rTT (duration of positive concentration curve) and TTP');
end

% calculate Peak Ratio
PR = 100*(1 - min(exp(-r2GdInBlood*TE*conc/1000), [], 4));

for s=1:options.nS
    for c=1:options.nC
        for r=1:options.nR
            if mask.data(r,c,s)>0 && max(abs(conc(r,c,s,:)))>0
                
                [BAT(r,c,s), TT(r,c,s), TTP(r,c,s), rTTP(r,c,s)] = dsc_calculate_BAT_TT_TTP_rTTP(conc(r,c,s,:),options.time,injectionTime);
%                 % define threshold based on peak value
%                 [peak, peakIdx] = max(conc(r,c,s,:));
%                 % first positive value
%                 bolusStart_idx = find(conc(r,c,s,:)>0.00001*peak, 1, 'first');
%                 % last positive value
%                 bolusEnd_idx = find(conc(r,c,s,:)>0.00001*peak, 1, 'last');
%                 
%                 % time between injection and bolus arrival (BAT)
%                 BAT(r,c,s) = options.time(bolusStart_idx) - injectionTime;
%                 % duration of bolus (Transit Time)
%                 TT(r,c,s) = options.time(bolusEnd_idx) - options.time(bolusStart_idx);
%                 % time between injection and peak (rTTP)
%                 rTTP(r,c,s) = options.time(peakIdx) - injectionTime;
%                 % time between bolus arrival and peak (TTP)
%                 TTP(r,c,s) = options.time(peakIdx) - options.time(bolusStart_idx);
                
            end
            
            % update waitbar
            if options.waitbar
                waitbar(((s-1)*options.nC*options.nR + (c-1)*options.nR + r)/(options.nR*options.nC*options.nS), hw_rMTT)
            end
            
        end
    end  
end
if options.waitbar
    delete(hw_rMTT);
end
end
