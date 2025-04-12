import os
import cv2
import pickle
import imutils
import argparse
import numpy as np
from glnet.utils.common_utils import _ex


def generate_sph_image(images, dataset_type, prespheredir):
    stitcher = Stitcher(dataset_type)
    sph_img = stitcher.stitch(images, prespheredir)
    sph_img = sph_img.astype(np.uint8)
    sph_img = cv2.cvtColor(sph_img, cv2.COLOR_BGR2RGB)
    sph_img = cv2.rotate(sph_img, cv2.ROTATE_180)
    return sph_img


def remove_the_blackborder(image):
    img = cv2.medianBlur(image, 5) 
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
 
    edges_y, edges_x = np.where(binary_image==255) ##h, w
    bottom = min(edges_y)             
    top = max(edges_y) 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    res_image = image[bottom:bottom+height, left:left+width]

    return res_image   


def XYZtoRC(image_meta, hits, H, W, cam_id):
    intrins = np.array(image_meta['K'])
    cams_T_body = np.array(image_meta['T'])
    # print(hits)
    K = intrins[cam_id]
    T = cams_T_body[cam_id]

    hits_c = np.matmul(T, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    x_im = hits_im[0, :] / (hits_im[2, :] + 1e-8)
    y_im = hits_im[1, :] / (hits_im[2, :] + 1e-8)
    z_im = hits_im[2, :]
    idx_infront = (z_im > 0) & (x_im > 0) & (x_im < W) & (y_im > 0) & (y_im < H)
    y_im = y_im[idx_infront]
    x_im = x_im[idx_infront]

    idx = (y_im.astype(int), x_im.astype(int))

    return idx, idx_infront


class Stitcher:
    def __init__(self, dataset_type, prespheredir):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)
        self.dataset_type = dataset_type
        self.prespheredir = prespheredir
        
        image_meta_path = _ex(os.path.join(prespheredir, 'image_meta.pkl'))
        with open(image_meta_path, 'rb') as handle:
            self.image_meta = pickle.load(handle)
        
        self.sphere_points_path = _ex(os.path.join(self.prespheredir, 'sphere_points.npy'))
        self.pix_path = _ex(os.path.join(self.prespheredir, 'pix.npy'))
    
    def stitch(self, images, scale=2):
        intrins = np.array(self.image_meta['K'])
        focal = intrins[3][1,1]
        sph_img, cv_img = self.spherical_projection(images, focal, scale)
        
        if scale == 2:
            if self.dataset_type == 'nclt':
                sph_img = cv_img[170:330,...] # for scale = 2
            elif self.dataset_type == 'oxford':
                sph_img = cv_img[200:400,...] # for scale = 2
        else:
            sph_img = remove_the_blackborder(cv_img)
        # cv2.imwrite('./sph.png', sph_img)

        return sph_img
    
    def sph_shift(self, sph_img, shift):
        sph_img = np.roll(sph_img, shift, axis=1)
        return sph_img
    
    def spherical_projection(self, imgs, f, scale=None) :
        num_cams = len(imgs)
        row, col = imgs[0].shape[:2]
        
        if scale is None:
            scale = num_cams
        outrow, outcol = scale * row, scale * col
        
        panoramic = np.zeros((outrow, outcol, 3))
        
        if os.path.exists(self.sphere_points_path) and os.path.exists(self.pix_path):
            points = np.load(self.sphere_points_path)
            pix = np.load(self.pix_path)
        else:
            points, pix_x, pix_y = self.generate_sphere_points(outrow, outcol, f)
            np.save(self.sphere_points_path, points)
            np.save(self.pix_path, pix)
        
        pix_x, pix_y = pix[1], pix[0]

        for cam_id in range(num_cams):
            idx, valid = XYZtoRC(self.image_meta, points, row, col, cam_id)

            rgb = imgs[cam_id][idx]
            valid_y, valid_x = pix_y[valid], pix_x[valid]

            if self.dataset_type == 'nclt':
                nonblack = np.where(rgb > 10, 1, 0)
                nonblack = np.sum(nonblack, axis=1)
                nonblack = np.where(nonblack > 2, True, False)
                panoramic[valid_y[nonblack], valid_x[nonblack], 0] = rgb[nonblack,0]
                panoramic[valid_y[nonblack], valid_x[nonblack], 1] = rgb[nonblack,1]
                panoramic[valid_y[nonblack], valid_x[nonblack], 2] = rgb[nonblack,2]
            elif self.dataset_type == 'oxford':
                panoramic[valid_y, valid_x, 0] = rgb[...,0]
                panoramic[valid_y, valid_x, 1] = rgb[...,1]
                panoramic[valid_y, valid_x, 2] = rgb[...,2]

            cv_img = cv2.merge([panoramic[...,0], panoramic[...,1], panoramic[...,2]])
            cv_img = cv_img.astype(np.uint8)

        return panoramic, cv_img

    def generate_sphere_points(self, outrow: int, outcol: int, R: float):
        """Generate sphere points and pixel mapping."""
        points = []
        pix_x, pix_y = [], []

        for y in range(outrow):
            for x in range(outcol):
                theta = -(2 * np.pi * x / (outcol - 1) - np.pi)
                phi = np.pi * y / (outrow - 1)

                globalZ = R * np.cos(phi)
                globalX = R * np.sin(phi) * np.cos(theta)
                globalY = R * np.sin(phi) * np.sin(theta)
                points.append([globalX, globalY, globalZ, 1])
                pix_x.append(x)
                pix_y.append(y)

        return np.asarray(points).T, np.asarray([pix_y, pix_x])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--dir', required=True,
                    help='path to the images dir')

    args = vars(args.parse_args())

    imagePaths = sorted(list(imutils.paths.list_images(args['dir'])))

    images = []

    # load the images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        images.append(image)

    # stitch the images together to create a panorama
    stitcher = Stitcher()
    result = stitcher.stitch(images, '~/Data/NCLT/')
    result = cv2.rotate(result, cv2.ROTATE_180)
    cv2.imwrite('./output.jpg', result)
