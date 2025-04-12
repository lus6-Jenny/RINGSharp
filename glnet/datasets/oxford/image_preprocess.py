import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import scipy.interpolate as interp
from scipy.ndimage import map_coordinates
from glnet.utils.common_utils import _ex

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'

class CameraModel:
    """Provides intrinsic parameters and undistortion LUT for a camera.

    Attributes:
        camera (str): Name of the camera.
        camera sensor (str): Name of the sensor on the camera for multi-sensor cameras.
        focal_length (tuple[float]): Focal length of the camera in horizontal and vertical axis, in pixels.
        principal_point (tuple[float]): Principal point of camera for pinhole projection model, in pixels.
        G_camera_image (:obj: `numpy.matrixlib.defmatrix.matrix`): Transform from image frame to camera frame.
        bilinear_lut (:obj: `numpy.ndarray`): Look-up table for undistortion of images, mapping pixels in an undistorted
            image to pixels in the distorted image

    """

    def __init__(self, models_dir, cam_name):
        """Loads a camera model from disk.

        Args:
            models_dir (str): directory containing camera model files.
            cam_name (str): camera model name.

        """
        self.camera = None
        self.camera_sensor = None
        self.focal_length = None
        self.principal_point = None
        self.G_camera_image = None
        self.bilinear_lut = None

        self.__load_intrinsics(models_dir, cam_name)
        self.__load_lut(models_dir, cam_name)

    def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.

        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative to camera frame.
            image_size (tuple[int]): dimensions of image in pixels

        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates of a point in pixels.
            numpy.array: array of depth values for points in image.

        Note:
            Number of output points m will be less than or equal to number of input points n, as points that do not
            project into the image are discarded.

        """
        if xyz.shape[0] == 3:
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(self.G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        in_img = [i for i in range(0, uv.shape[1])
                  if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]]

        return uv[:, in_img], np.ravel(xyzw[2, in_img])

    def undistort(self, image):
        """Undistorts an image.

        Args:
            image (:obj: `numpy.ndarray`): A distorted image. Must be demosaiced - ie. must be a 3-channel RGB image.

        Returns:
            numpy.ndarray: Undistorted version of image.

        Raises:
            ValueError: if image size does not match camera model.
            ValueError: if image only has a single channel.

        """
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]:
            raise ValueError('Incorrect image size for camera model')

        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))

        if len(image.shape) == 1:
            raise ValueError('Undistortion function only works with multi-channel images')

        undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

        return undistorted.astype(image.dtype)

    def __get_model_name(self, cam_name):
        self.camera = cam_name
        if self.camera == 'stereo/left':
            return 'stereo_wide_left'
        elif self.camera == 'stereo/right':
            return 'stereo_wide_right'
        elif self.camera == 'stereo/centre':
            return 'stereo_narrow_left'
        else:
            return self.camera

    def __load_intrinsics(self, models_dir, cam_name):
        model_name = self.__get_model_name(cam_name)
        intrinsics_path = os.path.join(models_dir, model_name + '.txt')

        with open(intrinsics_path) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            self.focal_length = (vals[0], vals[1])
            self.principal_point = (vals[2], vals[3])

            G_camera_image = []
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
            self.G_camera_image = np.array(G_camera_image)

    def __load_lut(self, models_dir, cam_name):
        model_name = self.__get_model_name(cam_name)
        lut_path = os.path.join(models_dir, model_name + '_distortion_lut.bin')

        lut = np.fromfile(lut_path, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.bilinear_lut = lut.transpose()


def load_image(image_path, model=None, debayer=True):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)

    if 'stereo' in camera:
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    if debayer:
        img = demosaic(img, pattern)
    # demosaic_im = Image.fromarray(img.astype('uint8')).convert('RGB')
    # demosaic_im = demosaic_im.save("./demosaic.jpg")

    if model:
        #print("Distorted!!")
        img = model.undistort(img)
        # undistort_im = Image.fromarray(img.astype('uint8')).convert('RGB')
        # undistort_im = undistort_im.save("./undistort.jpg")

    return np.array(img).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', type=str, default='~/Data/Oxford_radar')
    args = parser.parse_args()
    
    dataset_root = _ex(args.dataset_root)
    cam_name = ['stereo/centre', 'mono_left', 'mono_right', 'mono_rear']
    cam_save_name = ['stereo/centre_rect', 'mono_left_rect', 'mono_right_rect', 'mono_rear_rect']
    cam_models_dir = f'{dataset_root}/models/'
    extrinsics_dir = f'{dataset_root}/extrinsics/'
    
    sequences = ['2019-01-11-14-37-14', '2019-01-15-13-53-14', '2019-01-16-11-53-11', '2019-01-17-12-48-25', '2019-01-16-13-42-28']
    
    for sequence in sequences:
        for i in range(1,4):
            # i = 0
            cam = cam_name[i]
            cam_save = cam_save_name[i]
            camera_path = f'{dataset_root}/{sequence}/{cam}/'
            camera_save_path = f'{dataset_root}/{sequence}/{cam_save}/'
            
            model = CameraModel(cam_models_dir, cam)

            os.makedirs(camera_save_path, exist_ok=True)

            image_filenames = sorted(os.listdir(camera_path))

            for image_path in tqdm(image_filenames):
                image = load_image(camera_path + image_path, model)

                if model.camera == 'stereo/centre':
                    input_image = image[160:160+640,...]
                    input_image = cv2.resize(input_image, (640, 320))
                else:
                    input_image = image[200:200+512,...]
                    input_image = cv2.resize(input_image, (640, 320))

                image_name = image_path.split('.')[0]
                filename = camera_save_path + image_name + '.png'
                cv2.imwrite(filename, input_image)
