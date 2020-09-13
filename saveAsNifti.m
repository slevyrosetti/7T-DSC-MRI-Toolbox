function [] = saveAsNifti(oldNiiFname,data,oFname)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

%
newNii = load_untouch_nii(oldNiiFname);
% set new data
newNii.img = data;
% set the right 4th dimension if exists
if length(size(data))>3
    newNii.hdr.dime.dim(5) = size(data,4);
end
% set the right data type in header
if strcmp(class(data),'double') && ~ismember(newNii.hdr.dime.datatype, [16, 64])
    newNii.hdr.dime.datatype = 64;
%     newNii.hdr.dime.bitpix = 64;
end
% save as NIFTI
save_untouch_nii(newNii, oFname);
disp(['Saved nifti: ' oFname])

end

