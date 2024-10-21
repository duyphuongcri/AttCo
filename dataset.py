import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pathlib as plb
import os 
import SimpleITK as sitk
import scipy 
import torch 

class MedDataset(Dataset):
    """A class for fetching data samples.

    Parameters
    ----------
    paths_to_samples : list
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `mode == 'train'`, a path to
        a ground truth mask must be provided for each patient.
    transforms
        Transformations applied to each data sample.
    mode : str
        Must be `train` or `test`. If `train`, a ground truth mask is loaded using a path from `paths_to_samples` and
        added to a sample.
        If `test`, an additional information (an affine array), that describes the position of the image data
        in a reference space, is added to each data sample. Ground truth masks are not loaded in this mode.

    Returns
    -------
    dict
        A dictionary corresponding to a data sample.
        Keys:
            id : A patient's ID.
            input : A numpy array containing CT & PET images stacked along the last (4th) dimension.
            target : A numpy array containing a ground truth mask.
            affine : A numpy array with the position of the image data in a reference space (needed for resampling).
    """

    def __init__(self, path_root, listPatients, transforms=None, mode='train'):
        self.path_root=path_root
        self.patients_ID = listPatients #os.listdir(path_root)
        self.transforms = transforms
        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode


    def __len__(self):
        return len(self.patients_ID)

    def __getitem__(self, index):
        sample = dict()
        
        patientid = self.patients_ID[index]
        
        sample['id'] = patientid
        
        t1 = self.read_data(os.path.join(self.path_root, patientid, patientid+'_t1.nii.gz')).astype(np.float32)
        t1ce = self.read_data(os.path.join(self.path_root, patientid, patientid+'_t1ce.nii.gz')).astype(np.float32)
        flair = self.read_data(os.path.join(self.path_root, patientid, patientid+'_flair.nii.gz')).astype(np.float32)
        t2 = self.read_data(os.path.join(self.path_root, patientid, patientid+'_t2.nii.gz')).astype(np.float32)
        # print(sample['id'], t1.shape, t1ce.shape, flair.shape, t2.shape)
        img = np.stack([flair, t1, t1ce, t2], axis=-1)
        sample['input'] = img
        sample['input_ori'] = img.copy()

        mask = self.read_data(os.path.join(self.path_root, patientid, patientid+'_seg.nii.gz'))
        mask[mask == 4.] = 3.
        mask = mask*1.
        assert img.shape[:-1] == mask.shape, \
            f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

        sample['target'] = mask
        sample['target_ori'] = mask.copy()

        if self.transforms:
            sample = self.transforms(sample)

        ###################
        sample['wt'] = torch.zeros_like(sample['target'])
        sample['tc'] = torch.zeros_like(sample['target'])
        sample['et'] = torch.zeros_like(sample['target'])
        sample['wt'][sample['target'] > 0]= 1.
        sample['tc'][sample['target'] == 1]= 1.
        sample['tc'][sample['target'] == 3]= 1.
        sample['et'][sample['target'] == 3]= 1.
        sample['wt'] = sample['wt'].unsqueeze(0).float()
        sample['tc'] = sample['tc'].unsqueeze(0).float()
        sample['et'] = sample['et'].unsqueeze(0).float()
        return sample

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        # if return_numpy:
        #     return nib.load(str(path_to_nifti)).get_fdata()
        # return nib.load(str(path_to_nifti))
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))

import pathlib as plb
def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)
        
        #dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs

class AutoPETDataset(Dataset): ## AutoPET Challenge
    """A class for fetching data samples.

    Parameters
    ----------
    paths_to_samples : list
        A list wherein each element is a tuple with two (three) `pathlib.Path` objects for a single patient.
        The first one is the path to the CT image, the second one - to the PET image. If `mode == 'train'`, a path to
        a ground truth mask must be provided for each patient.
    transforms
        Transformations applied to each data sample.
    mode : str
        Must be `train` or `test`. If `train`, a ground truth mask is loaded using a path from `paths_to_samples` and
        added to a sample.
        If `test`, an additional information (an affine array), that describes the position of the image data
        in a reference space, is added to each data sample. Ground truth masks are not loaded in this mode.

    Returns
    -------
    dict
        A dictionary corresponding to a data sample.
        Keys:
            id : A patient's ID.
            input : A numpy array containing CT & PET images stacked along the last (4th) dimension.
            target : A numpy array containing a ground truth mask.
            affine : A numpy array with the position of the image data in a reference space (needed for resampling).
    """

    def __init__(self, path_root, listPatients, transforms=None, mode='train'):
        self.path_root=path_root
        # self.paths_to_samples = listPatients
        self.paths_to_samples = [fn.replace("/mnt/sdc/data/FDG-PET-CT-Lesions/PositiveCases_preprocessed", path_root) for fn in listPatients]
        self.transforms = transforms
        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode
        # if mode == 'train':
        #     self.num_of_seqs = len(paths_to_samples[0]) - 1
        # else:
        #     self.num_of_seqs = len(paths_to_samples[0])

    def __len__(self):
        return len(self.paths_to_samples)

    def __getitem__(self, index):
        sample = dict()

        id_ = self.paths_to_samples[index]#.parent.stem
        sample['id'] = id_
        if self.mode == "train":
            ct = np.load(os.path.join(self.paths_to_samples[index], 'CTres.npy'))
            pet = np.load(os.path.join(self.paths_to_samples[index], 'SUV.npy'))

            img = np.stack([ct, pet], axis=-1)
            sample['input'] = img

            mask = np.load(os.path.join(self.paths_to_samples[index], 'SEG.npy'))
            mask = np.expand_dims(mask, axis=3)
            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask

        elif self.mode == "val":
            ct_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'CTres.nii.gz')))
            pet_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'SUV.nii.gz')))
            mask_ori = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.paths_to_samples[index], 'SEG.nii.gz')))

            sample['ct_ori'] = np.flip(ct_ori, axis=0).copy()
            sample['pet_ori'] = np.flip(pet_ori, axis=0).copy()
            sample['mask_ori'] = np.flip(mask_ori, axis=0).copy()
    
            ct = np.load(os.path.join(self.paths_to_samples[index], 'CTres.npy'))
            pet = np.load(os.path.join(self.paths_to_samples[index], 'SUV.npy'))

            img = np.stack([ct, pet], axis=-1)
            sample['input'] = img

            mask = np.load(os.path.join(self.paths_to_samples[index], 'SEG.npy'))
            mask = np.expand_dims(mask, axis=3)
            assert img.shape[:-1] == mask.shape[:-1], \
                f"Shape mismatch for the image with the shape {img.shape} and the mask with the shape {mask.shape}."

            sample['target'] = mask
    
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    @staticmethod
    def read_data(path_to_nifti, return_numpy=True):
        """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
        # if return_numpy:
        #     return nib.load(str(path_to_nifti)).get_fdata()
        # return nib.load(str(path_to_nifti))
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_to_nifti)))