function [rCBV, rCBF]=dsc_rCBV_rCBF(conc,mask,options)
%Funzione del pacchetto DSC_mri - DSC_mri_cbv
%Autore: Castellaro Marco - Universit� di Padova - DEI
%
%Calcola le mappe parametriche di Cerebral Blood Volume per un soggetto
%
%Parametri in ingresso: conc (Matrice 4D) che contiene gli andamenti delle
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


if options.display > 0
    disp('   rCBV (area under the curve) & rCBF (dC/dt)');
end

rCBV=zeros(options.nR,options.nC,options.nS);
rCBF=zeros(options.nR,options.nC,options.nS);

if options.waitbar
    hw_cbv=waitbar(0,'Calculating rCBV (area under the curve) & rCBF (dC/dt)');
end
for s=1:options.nS
    
    % relative Blood Volume (Area Under the Curve)
    rCBV(:,:,s)=(options.par.kh/options.par.rho)*mask.data(:,:,s).* trapz(options.time,conc(:,:,s,:),4);
    
    % relative Blood Flow (max of dC/dt)
    dC = diff(squeeze(conc(:,:,s,:)), 1, 3);
    dt = repmat(diff(options.time)', 1, options.nC, options.nR);
    dt = permute(dt, [3,2,1]);
    rCBF(:,:,s)=(options.par.kh/options.par.rho)*mask.data(:,:,s).* max(dC./dt,[],3);
    
    if options.waitbar
        waitbar(s/options.nS,hw_cbv)
    end
end
if options.waitbar
    delete(hw_cbv);
end
end
