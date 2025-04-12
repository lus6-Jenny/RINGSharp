# !/usr/bin/python
#
# Demonstrates how to project velodyne points to camera imagery. Requires a binary
# velodyne sync file, undistorted image, and assumes that the calibration files are
# in the directory.
#
# To use:
#
#    python project_vel_to_cam_rot.py vel img cam_num
#    python project_vel_to_cam_rot.py /home/lusha/Data/NCLT/2012-02-04/velodyne_sync/1328389911665289.bin /home/lusha/Data/NCLT/2012-02-04/lb3_u_s_384/Cam1/1328389911665289.jpg 1
#       vel:  The velodyne binary file (timestamp.bin)
#       img:  The undistorted image (timestamp.tiff)
#   cam_num:  The index (0 through 5) of the camera

import os
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from glnet.utils.common_utils import _ex

def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_vel_hits(filename):
    f_bin = open(filename, "rb")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == b'': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)
        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)

        # Load in homogenous
        hits += [[x, y, z, 1]]

    f_bin.close()
    hits = np.asarray(hits)

    print("height median:", np.median(hits[:,2]))
    print("max height:", np.max(hits[:,2]))
    print("min height:", np.min(hits[:,2]))    
    # mask = hits[:, 2] < -0.3
    # hits = hits[mask]

    return hits.transpose()

def ssc_to_homo(ssc):
    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def project_vel_to_cam(hits, cam_num):
    # Load camera parameters    
    image_meta_path = _ex('~/Data/NCLT/image_meta.pkl')
    with open(image_meta_path, 'rb') as handle:
        image_meta = pickle.load(handle)
    intrins = np.array(image_meta['K'])
    cams_T_body = np.array(image_meta['T'])
    K = intrins[cam_num-1]
    T_camNormal_body = cams_T_body[cam_num-1]
    hits_c = np.matmul(T_camNormal_body, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    return hits_im

def project_vel_to_cam_oxford(hits, cam_num):
    # Load camera parameters
    image_meta_path = _ex('~/Data/Oxford_radar/image_meta.pkl')
    with open(image_meta_path, 'rb') as handle:
        image_meta = pickle.load(handle)
    intrins = np.array(image_meta['K'])
    cams_T_body = np.array(image_meta['T'])
    K = intrins[cam_num]
    T_camNormal_body = cams_T_body[cam_num]
    hits_c = np.matmul(T_camNormal_body, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    return hits_im

def load_im_file_for_generate(filename):
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE) # ROTATE_90_COUNTERCLOCKWISE
    return input_image

def main(args):
    if len(args) < 4:
        print("""Incorrect usage. To use:
            python project_vel_to_cam.py vel img cam_num
            vel:  The velodyne binary file (timestamp.bin)
            img:  The undistorted image (timestamp.tiff)
            cam_num:  The index (0 through 5) of the camera
            """)
        return 1

    # image = mpimg.imread(args[2])
    image = load_im_file_for_generate(args[2])
    cam_num = int(args[3])
    # Load velodyne points
    hits_body = load_vel_hits(args[1])
    print(hits_body.shape)

    hits_image = project_vel_to_cam(hits_body, cam_num)
    print(hits_image.shape)

    x_im = hits_image[0, :]/hits_image[2, :]
    y_im = hits_image[1, :]/hits_image[2, :]
    z_im = hits_image[2, :]

    idx_infront = (z_im > 0) & (x_im > 0) & (x_im < 384) & (y_im > 0) & (y_im < 224)
    # idx_infront = (z_im > 0) & (x_im > 0) & (x_im < 1232) & (y_im > 0) & (y_im < 1616)

    hits_body_image2 = hits_body.transpose()[idx_infront]

    x_im = x_im[idx_infront]
    y_im = y_im[idx_infront]
    z_im = z_im[idx_infront]

    x_im_int = x_im.astype(np.int32)
    y_im_int = y_im.astype(np.int32)
    z_im_int = z_im.astype(np.int32)
    print(x_im_int.shape)
    image = np.asarray(image)

    plt.figure(1)
    plt.imshow(image)
    plt.scatter(x_im, y_im, c=z_im%20.0/20.0, cmap='jet', alpha=0.1, s=1)
    # plt.xlim(0, 240)
    # plt.ylim(0, 360)
    plt.show()
    plt.savefig('./proj_rect.jpg')

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
