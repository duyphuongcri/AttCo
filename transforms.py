import random
import numpy as np
import torch
import scipy
from skimage.transform import rotate
from scipy.interpolate import RegularGridInterpolator 
from scipy.ndimage import gaussian_filter

class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class ToTensor:
    def __init__(self, mode='train', data='brats'):
        self.data=data
        if mode not in ['train', 'test']:
            raise ValueError(f"Argument 'mode' must be 'train' or 'test'. Received {mode}")
        self.mode = mode

    def __call__(self, sample):
        if self.mode == 'train' or self.mode == 'val':
            img, mask = sample['input'], sample['target']
            img = np.transpose(img, axes=[3, 0, 1, 2])
            if self.data != 'brats':
                mask = np.transpose(mask, axes=[3, 0, 1, 2])
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            sample['input'], sample['target'] = img, mask

        return sample

class GaussianNoise:
    """
    Elastic deformation of images as described in
    Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual
    Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62

    Modified to take 3D inputs
    Deforms both the image and corresponding label file
    image linear/trilinear interpolated

    Label volumes nearest neighbour interpolated
    """
    def __init__(self, p=0.5, alpha=4, sigma=35, bg_val=0.1):
        self.p = p
        self.alpha=alpha
        self.sigma=sigma
        self.bg_val=bg_val

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']
            shape = img.shape[:3]
            # Define coordinate system
            coords = np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2])

            for i in range(img.shape[3]):

                # Initialize interpolators
                im_intrps = RegularGridInterpolator(coords, img[:,:,:,i],
                                                                method="linear",
                                                                bounds_error=False,
                                                                fill_value=self.bg_val)
                # Get random elastic deformations
                dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma,
                                        mode="constant", cval=0.) * self.alpha
                dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma,
                                        mode="constant", cval=0.) * self.alpha
                dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma,
                                        mode="constant", cval=0.) * self.alpha

                # Define sample points
                z, x, y = np.mgrid[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]]
                indices = np.reshape(z + dz, (-1, 1)), \
                            np.reshape(x + dx, (-1, 1)), \
                            np.reshape(y + dy, (-1, 1))

                # Interpolate 3D image image
                # image = np.empty(shape=shape)
                img[:,:,:,i] = im_intrps(indices).reshape(shape)

            # Interpolate labels
            lab_intrp = RegularGridInterpolator(coords, mask[:,:,:,0],
                                                method="nearest",
                                                bounds_error=False,
                                                fill_value=0)
            mask[:,:,:,0] = lab_intrp(indices).reshape(shape)
            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample


class Mirroring:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            n_axes = random.randint(0, 3)
            random_axes = random.sample(range(3), n_axes)

            img = np.flip(img, axis=tuple(random_axes))
            mask = np.flip(mask, axis=tuple(random_axes))

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample


class RandZoom:
    def __init__(self, p=0.5, min_percentage=0.85, max_percentage=1.15):
        self.p = p
        self.min_percentage=min_percentage
        self.max_percentage=max_percentage

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']
            z = np.random.sample() *(self.max_percentage-self.min_percentage) + self.min_percentage
            zoom_matrix = np.array([[z, 0, 0, 0],
                                    [0, z, 0, 0],
                                    [0, 0, z, 0],
                                    [0, 0, 0, 1]])
            img = scipy.ndimage.interpolation.affine_transform(img, zoom_matrix)
            mask = scipy.ndimage.interpolation.affine_transform(mask, zoom_matrix)

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample


class Resample:
    def __init__(self, scale_factor=(0.5, 0.5, 0.5)):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            img = scipy.ndimage.interpolation.zoom(img, self.scale_factor, mode="nearest")
            mask = scipy.ndimage.interpolation.zoom(mask, self.scale_factor, mode="nearest")

            sample['input'], sample['target'] = img.copy(), mask.copy()

        return sample

class NormalizeIntensity:

    def __call__(self, sample):
        img = sample['input']
        img1 = img[:, :, :, 0]
        img2 = img[:, :, :, 1]
        img3 = img[:, :, :, 2]
        img4 = img[:, :, :, 3]
       
        img1 = self.standadize_nonzeros(img1)
        img2 = self.standadize_nonzeros(img2)
        img3 = self.standadize_nonzeros(img3)
        img4 = self.standadize_nonzeros(img4)

        sample['input'] = np.stack((img1, img2, img3, img4), axis=-1)
        return sample

    @staticmethod
    def normalize_minmax(image):
        norm_img = (image - image.min()) / (image.max()-image.min())
        return norm_img
    @staticmethod
    def standadize_nonzeros(image):
        img_nonzeros = image[image!=0]
        norm_img = (image - img_nonzeros.mean()) / img_nonzeros.std()
        return norm_img

class NormalizeIntensity_AutoPET:

    def __call__(self, sample):
        img = sample['input']
        img[:, :, :, 0] = self.normalize_ct(img[:, :, :, 0])
        img[:, :, :, 1] = self.normalize_pt(img[:, :, :, 1])

        sample['input'] = img
        return sample

    @staticmethod
    def normalize_ct(img):
        norm_img = np.clip(img, -1024, 1024) / 1024
        return norm_img

    @staticmethod
    def normalize_pt(img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-3)

class RandomCrop:
    def __init__(self, margin=(16, 16, 16), target_size=(128,128,128), original_size=(155,240,240)):
        self.margin=margin
        self.target_size=target_size
        self.original_size = original_size
        
    def __call__(self, sample):
        img, mask = sample['input'], sample['target']

        indexes = np.where(mask != 0)
        z_centroid = (indexes[0].min() + indexes[0].max()) // 2
        y_centroid = (indexes[1].min() + indexes[1].max()) // 2
        x_centroid = (indexes[2].min() + indexes[2].max()) // 2

        point = [0, 0, 0]
        point[0] = z_centroid - self.target_size[0]//2 + random.randint(-self.margin[0], self.margin[0])
        point[1] = y_centroid - self.target_size[1]//2 + random.randint(-self.margin[1], self.margin[1])
        point[2] = x_centroid - self.target_size[2]//2 + random.randint(-self.margin[2], self.margin[2])

        for i in range(3):
            if point[i] < 0:
                point[i] = 0
            elif point[i] + self.target_size[i] > self.original_size[i]:
                point[i] = self.original_size[i] - self.target_size[i]

        d, h, w = self.target_size
        img_cropped = img[point[0]:point[0]+ d, point[1]:point[1]+ h, point[2]:point[2]+ w, :]
        mask_cropped = mask[point[0]:point[0]+ d, point[1]:point[1]+ h, point[2]:point[2]+ w]
        sample['input'], sample['target'] = img_cropped, mask_cropped
        sample['point'] = np.array(point)
        return sample

class RandomRotation:
    def __init__(self, p=0.5, angle_range=[5, 15]):
        self.p = p
        self.angle_range = angle_range

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['input'], sample['target']

            num_of_seqs = img.shape[-1]
            n_axes = random.randint(1, 3)
            random_axes = random.sample([0, 1, 2], n_axes)

            for axis in random_axes:

                angle = random.randrange(*self.angle_range)
                angle = -angle if random.random() < 0.5 else angle

                for i in range(num_of_seqs):
                    img[:, :, :, i] = RandomRotation.rotate_3d_along_axis(img[:, :, :, i], angle, axis, 1)

                mask[:, :, :, 0] = RandomRotation.rotate_3d_along_axis(mask[:, :, :, 0], angle, axis, 0)

            sample['input'], sample['target'] = img, mask
        return sample

    @staticmethod
    def rotate_3d_along_axis(img, angle, axis, order):

        if axis == 0:
            rot_img = rotate(img, angle, order=order, preserve_range=True)

        if axis == 1:
            rot_img = np.transpose(img, axes=(1, 2, 0))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(2, 0, 1))

        if axis == 2:
            rot_img = np.transpose(img, axes=(2, 0, 1))
            rot_img = rotate(rot_img, angle, order=order, preserve_range=True)
            rot_img = np.transpose(rot_img, axes=(1, 2, 0))

        return rot_img




class ExtractPatch:
    """Extracts a patch of a given size from an image (4D numpy array)."""

    def __init__(self, patch_size, p_tumor=0.5):
        self.patch_size = patch_size  # without channel dimension!
        self.p_tumor = p_tumor  # probs to extract a patch with a tumor

    def __call__(self, sample):
        img = sample['input']
        mask = sample['target']

        assert all(x <= y for x, y in zip(self.patch_size, img.shape[:-1])), \
            f"Cannot extract the patch with the shape {self.patch_size} from  " \
                f"the image with the shape {img.shape}."

        # patch_size components:
        ps_x, ps_y, ps_z = self.patch_size

        if random.random() < self.p_tumor:
            # coordinates of the tumor's center:
            xs, ys, zs, _ = np.where(mask != 0)
            tumor_center_x = np.min(xs) + (np.max(xs) - np.min(xs)) // 2
            tumor_center_y = np.min(ys) + (np.max(ys) - np.min(ys)) // 2
            tumor_center_z = np.min(zs) + (np.max(zs) - np.min(zs)) // 2

            # compute the origin of the patch:
            patch_org_x = random.randint(tumor_center_x - ps_x, tumor_center_x)
            patch_org_x = np.clip(patch_org_x, 0, img.shape[0] - ps_x)

            patch_org_y = random.randint(tumor_center_y - ps_y, tumor_center_y)
            patch_org_y = np.clip(patch_org_y, 0, img.shape[1] - ps_y)

            patch_org_z = random.randint(tumor_center_z - ps_z, tumor_center_z)
            patch_org_z = np.clip(patch_org_z, 0, img.shape[2] - ps_z)
        else:
            patch_org_x = random.randint(0, img.shape[0] - ps_x)
            patch_org_y = random.randint(0, img.shape[1] - ps_y)
            patch_org_z = random.randint(0, img.shape[2] - ps_z)

        # extract the patch:
        patch_img = img[patch_org_x: patch_org_x + ps_x,
                    patch_org_y: patch_org_y + ps_y,
                    patch_org_z: patch_org_z + ps_z,
                    :].copy()

        patch_mask = mask[patch_org_x: patch_org_x + ps_x,
                     patch_org_y: patch_org_y + ps_y,
                     patch_org_z: patch_org_z + ps_z,
                     :].copy()

        assert patch_img.shape[:-1] == self.patch_size, \
            f"Shape mismatch for the patch with the shape {patch_img.shape[:-1]}, " \
                f"whereas the required shape is {self.patch_size}."

        sample['input'] = patch_img
        sample['target'] = patch_mask

        return sample


