import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef
import os
import random
import yaml
import pdb

def ResampleCMRImage(imImage, imLabel, save_path, patient_name, count, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()


    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))
    
    
    re_img = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab = ResampleLabelToRef(imLabel, re_img, interp=sitk.sitkNearestNeighbor)


    sitk.WriteImage(re_img, '%s/%s_%d.nii.gz'%(save_path, patient_name, count))
    sitk.WriteImage(re_lab, '%s/%s_%d_gt.nii.gz'%(save_path, patient_name, count))



if __name__ == '__main__':


    src_path = '/data/groups/beets-tan/l.cai/nnUNet_raw_data_base/Dataset040_tumouronly'
    tgt_path = '/data/groups/beets-tan/l.cai/cbim/cbim_Data/zuyd392d'




    patient_list = sorted(os.listdir(os.path.join(src_path, 'labelsTr')))#list(range(1, 101))

    name_list = patient_list#[]\
    name_list = [i.rstrip('.nii.gz') for i in name_list]

    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    #os.chdir(src_path)
    for name in name_list:#os.listdir('.'):
        #os.chdir(name)
        
        count = 0
        #for i in os.listdir('.'):
        #    if 'gt' in i:
        #        tmp = i.split('_')
        #        img_name = tmp[0] + '_' + tmp[1]
        #        patient_name = tmp[0]

        img = sitk.ReadImage(os.path.join(src_path, 'imagesTr', f'{name}_0000.nii.gz'))
        lab = sitk.ReadImage(os.path.join(src_path, 'labelsTr', f'{name}.nii.gz'))
        ResampleCMRImage(img, lab, tgt_path, name, count, (0.7291666865348816,0.7291666865348816))
        count += 1
        print(name, 'done')

        #os.chdir('..')


