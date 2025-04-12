import argparse
import os
import cv2
import copy
import numpy as np
import torch
import pickle
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation
from glnet.models.extractor_matcher.superpoint import SuperPoint
from glnet.models.extractor_matcher.matching import Matching
from glnet.models.extractor_matcher.utils import scale_intrinsics, process_resize, estimate_pose, make_matching_plot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def convert_to_tensor(data):
    if isinstance(data, list):
        positions = np.array([(kp.pt[0], kp.pt[1]) for kp in data], dtype=np.float32)
        return torch.tensor(positions)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise ValueError("Unsupported data type. Input must be a list or NumPy array.")
    
    
def mat_to_euler_angles(R):
    # Extract Euler angles (roll, pitch, yaw)
    r = Rotation.from_matrix(R)
    quaternion = r.as_quat()
    # Convert quaternion to Euler angles
    r_euler = Rotation.from_quat(quaternion)
    euler_angles = r_euler.as_euler('xyz')
    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]
    
    return roll, pitch, yaw


def preprocess_image(dataset_type, img_path, resize=[-1], resize_float=True):
    imgs = []
    imgs_gray = []
    imgs_norm = []     
    if isinstance(img_path, str):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if dataset_type == 'nclt':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        w, h = img.shape[1], img.shape[0]
        w_new, h_new = process_resize(w, h, resize)
        scales = (float(w) / float(w_new), float(h) / float(h_new))
        
        if resize_float:
            img = cv2.resize(img.astype('float32'), (w_new, h_new))
        else:
            img = cv2.resize(img, (w_new, h_new)).astype('float32')
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_norm = torch.from_numpy(img_gray / 255.).float()[None, None].to(device)
        img = img[:, :, ::-1]  # BGR to RGB
        return img, img_gray, img_norm, scales
    
    elif isinstance(img_path, (list, np.ndarray)):
        for file in img_path:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            if dataset_type == 'nclt':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            w, h = img.shape[1], img.shape[0]
            w_new, h_new = process_resize(w, h, resize)
            scales = (float(w) / float(w_new), float(h) / float(h_new))            

            if resize_float:
                img = cv2.resize(img.astype('float32'), (w_new, h_new))
            else:
                img = cv2.resize(img, (w_new, h_new)).astype('float32')
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            
            img_norm = torch.from_numpy(img_gray / 255.).float()[None, None].to(device)
            img = img[:, :, ::-1]  # BGR to RGB
            imgs.append(img)
            imgs_gray.append(img_gray)
            imgs_norm.append(img_norm)
        return imgs, imgs_gray, imgs_norm, scales
    else:
        raise ValueError('Image file path must be string, list or np.ndarray type')


def draw_keypoints(img, pts, color=(0, 255, 0)):
    out = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
      pt1 = (int(round(pt[0])), int(round(pt[1])))
      cv2.circle(out, pt1, 1, color, -1, lineType=16)

    return out


def draw_matches(rgb1, rgb2, match_pairs, lw = 0.5, color='g', if_fig=True,
                filename='matches.png', show=False):
    '''
    :param rgb1:
        image1
        numpy (H, W)
    :param rgb2:
        image2
        numpy (H, W)
    :param match_pairs:
        numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
    :return:
        None
    '''
    from matplotlib import pyplot as plt

    h1, w1 = rgb1.shape[:2]
    h2, w2 = rgb2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
    canvas[:h1, :w1] = rgb1[:,:,np.newaxis]
    canvas[:h2, w1:] = rgb2[:,:,np.newaxis]
    # fig = plt.figure(frameon=False)
    if if_fig:
        fig = plt.figure(figsize=(15,5))
    plt.axis("off")
    plt.imshow(canvas, zorder=1)

    xs = match_pairs[:, [0, 2]]
    xs[:, 1] += w1
    ys = match_pairs[:, [1, 3]]

    alpha = 1
    sf = 5
    # lw = 0.5
    # markersize = 1
    markersize = 2

    plt.plot(
        xs.T, ys.T,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        marker='o',
        markersize=markersize,
        fillstyle='none',
        color=color,
        zorder=2,
        # color=[0.0, 0.8, 0.0],
    );
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    print('#Matches = {}'.format(len(match_pairs)))
    if show:
        plt.show()


# from utils.draw import draw_matches_cv
def draw_matches_cv(img1, kp1, img2, kp2, matches, inliers):
    # keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in kp1]
    # keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in kp2]
    keypoints1 = kp1
    keypoints2 = kp2
    matches = np.array(matches)[inliers.astype(bool)].tolist()
    def to3dim(img):
        if img.ndim == 2:
            img = np.dstack((img, img, img))
        return img
    img1 = to3dim(img1).astype(np.uint8)
    img2 = to3dim(img2).astype(np.uint8)
    
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                    None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))
    return matched_img


def extract_SIFT_keypoints_and_descriptors(gray_img):
    gray_img = gray_img.astype('uint8')
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    # Trim the keypoints near to image edges
    if len(kp) != 0:
        kp = cv2.KeyPoint_convert(kp)
        img_width = gray_img.shape[1]
        valid_keypoints_mask = (kp[:, 0] >= 16) & (kp[:, 0] <= img_width - 40)
        kp = kp[valid_keypoints_mask]
        desc = desc[valid_keypoints_mask]
        kp = [cv2.KeyPoint(p[0], p[1], 1) for p in kp]

    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, img_width=384,
                                                 keep_k_points=1000):
    keypoint_map = keypoint_map.T
    # Filter out the negative prob
    keypoint_map = keypoint_map[keypoint_map[:, 2] > 0]
    # Sort the prob
    sorted_prob = np.argsort(keypoint_map[:, 2])
    start = min(keep_k_points, keypoint_map.shape[0])
    keypoints = keypoint_map[sorted_prob[-start:], :2]
    desc = descriptor_map.T[sorted_prob[-start:], :]

    # Trim the keypoints near to image edges
    # print('before min & max x coords', np.min(keypoints[:,0]), np.max(keypoints[:,0]))
    # print('before min & max y coords', np.min(keypoints[:,1]), np.max(keypoints[:,1]))
    valid_keypoints_mask = (keypoints[:, 0] >= 16) & (keypoints[:, 0] <= img_width - 40)
    keypoints = keypoints[valid_keypoints_mask]
    desc = desc[valid_keypoints_mask]
    # print('after min & max x coords', np.min(keypoints[:,0]), np.max(keypoints[:,0]))
    # print('after min & max y coords', np.min(keypoints[:,1]), np.max(keypoints[:,1]))

    keypoints = [cv2.KeyPoint(p[0], p[1], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors_nn(kp1, desc1, kp2, desc2, threshold=0.75):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    distances = [m.distance for m in matches]
    max_distance = max(distances) if distances else 1.0    
    mconf = [1.0 - m.distance / max_distance for m in matches]
    
    return m_kp1, m_kp2, matches, mconf


def match_descriptors(model, data):
    pred = model(data)
    has_none_value = any(value is None for value in pred.values())
    if not has_none_value:
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']            
        # Keep the matching keypoints
        valid = matches > -1
        if True in valid:
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            matches = matches[valid]    
            mconf = conf[valid]        
        else:
            mkpts0 = mkpts1 = matches = mconf = None
    else:
        kpts0 = kpts1 = mkpts0 = mkpts1 = matches = mconf = None

    return kpts0, kpts1, mkpts0, mkpts1, matches, mconf


def compute_homography(matched_kp1, matched_kp2, K):
    if isinstance(matched_kp1, list): 
        matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
        matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    elif isinstance(matched_kp1, torch.Tensor):
        matched_pts1 = matched_pts1.detach().cpu().numpy()
        matched_pts2 = matched_pts2.detach().cpu().numpy()
    else:
        matched_pts1 = matched_kp1
        matched_pts2 = matched_kp2
    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1,
                                    matched_pts2,
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    
    E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, K, cv2.RANSAC, 0.999, 1.0)
    pass_count, R, t, mask = cv2.recoverPose(E, matched_pts1, matched_pts2, K)
    mask = mask.flatten()
    
    return R, t, inliers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography between two imgs with the SuperPoint feature matches.')
    parser.add_argument('img1_path', type=str)
    parser.add_argument('img2_path', type=str)            
    parser.add_argument(
        '--cam1_idx', type=int, default=1, help='Camera index of the first image input in mutli camera setting')    
    parser.add_argument(
        '--cam2_idx', type=int, default=1, help='Camera index of the second image input in mutli camera setting')        
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')      
    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')                      
    args = parser.parse_args()
    
    dataset = 'nclt'
    image_meta_path = os.path.expanduser("~/Data/NCLT/image_meta.pkl")
    with open(image_meta_path, 'rb') as handle:
        image_meta = pickle.load(handle)
    K = np.array(image_meta['K'])

    config = {
        'sift': {
            'grayscale': True,
            'resize_max': 1600,
            'trim_edges': [16, -40],
        },
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
            'weights': args.superglue,
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_threshold,
        },
        'nn-ratio': {
            'ratio_threshold': 0.8, 
            'distance_threshold': 0.7,
        },
        'nn-dis': {
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
        'nn-mutual': {
            'do_mutual_check': True,       
        }
    }
    matching = Matching(config).eval().to(device)

    img1_file = args.img1_path
    img2_file = args.img2_path
    cam1_idx = args.cam1_idx
    cam2_idx = args.cam2_idx
    resize = args.resize
    resize_float = args.resize_float
    max_keypoints = args.max_keypoints
    K1 = K[cam1_idx]
    K2 = K[cam2_idx]
    
    # Image Processing
    img1_orig, img1_gray, img1, scales1 = preprocess_image(dataset, img1_file, resize, resize_float)
    img2_orig, img2_gray, img2, scales2 = preprocess_image(dataset, img2_file, resize, resize_float)
    print('original image shape', img1_orig.shape)
    print('gray image shape', img1_gray.shape)

    # Scale the intrinsics to resized image
    K1 = scale_intrinsics(K1, scales1)
    K2 = scale_intrinsics(K2, scales2)
    
    # SIFT + NN
    data = {'image0': img1_gray, 'image1': img2_gray, 'extractor': 'sift', 'matcher': 'nn-ratio'}   
    # SuperPoint + NN
    data = {'image0': img1, 'image1': img2, 'extractor': 'superpoint', 'matcher': 'nn-ratio'}  
    # # SuperPoint + SuperGlue
    data = {'image0': img1, 'image1': img2, 'extractor': 'superpoint', 'matcher': 'superglue'}  
    
    #  ------ Feature Extraction + Feature Matching ------
    extractor = data.get('extractor', 'superpoint') 
    matcher = data.get('matcher', 'superglue') 
    kp1, kp2, m_kp1, m_kp2, matches, conf = match_descriptors(matching, data)   
    print('keypoints shape', kp1.shape, kp2.shape)
    # R, t, inliers = compute_homography(m_kp1, m_kp2, K1)
    thresh = 1.
    R, t, inliers = estimate_pose(m_kp1, m_kp2, K1, K2, thresh)
    yaw, pitch, roll = mat_to_euler_angles(R)
    exp_name = f'{extractor}_{matcher}'
    print(f"{exp_name} R: ", R)
    print(f"{exp_name} t: ", t)
    print(f"{exp_name} yaw: ", yaw)
    print(f"{exp_name} pitch: ", pitch)
    print(f"{exp_name} roll: ", roll)    
    print(f"{exp_name} keypoints shape", kp1.shape, m_kp1.shape)
    print(f"{exp_name} inliers: ", np.sum(inliers))
    
    if args.viz:
        viz_path = f"{exp_name}_matches.jpg"
        # Visualize the matches
        color = cm.jet(conf)
        text = [
            exp_name,
            'Keypoints: {}:{}'.format(len(kp1), len(kp2)),
            'Matches: {}'.format(len(m_kp1)),
        ]

        # Display extra parameter info
        if extractor == 'superpoint' and matcher == 'superglue':
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]
        else:
            small_text = []

        make_matching_plot(
            img1_gray, img2_gray, kp1, kp2, m_kp1, m_kp2, color,
            text, viz_path, show_keypoints=True,
            fast_viz=False, opencv_display=False, opencv_title='Matches', small_text=small_text) 