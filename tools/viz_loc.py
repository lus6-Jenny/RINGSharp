# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified from OrientNet

import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

import matplotlib
matplotlib.rcParams['font.family'] = 'Arial' # 'Helvetica', 'Times New Roman'

def transform_point_cloud(a, T):
    homogeneous_a = np.hstack((a, np.ones((a.shape[0], 1))))
    transformed_a = np.dot(T, homogeneous_a.T).T

    return transformed_a[:, :3]


def plot_point_cloud(pc, s=0.3, c='g', alpha=0.5, save_path=None, transpose=False, **kwargs):
    ax = plt.gca()
    if transpose:
        ax.scatter(pc[:, 1], pc[:, 0], s=s, c=c, alpha=alpha, **kwargs)
    else:
        ax.scatter(pc[:, 0], pc[:, 1], s=s, c=c, alpha=alpha, **kwargs)
    # ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    for spine in ax.spines.values():  # remove frame
        spine.set_visible(False)
    
    if save_path is not None:
        save_plot(save_path, dpi=600)
    plt.close()
    

def plot_matched_point_clouds(source_pc, target_pc, s=0.3, source_color='g', target_color='b', alpha=0.5, save_path=None, transpose=False, **kwargs):
    """
    Plots a pair of matched point clouds.

    Parameters:
        source_pc (array-like): The source point cloud, an Nx2 or Nx3 array.
        target_pc (array-like): The target point cloud, an Nx2 or Nx3 array.
        s (float): The size of each point.
        source_color (str): Color for the source point cloud.
        target_color (str): Color for the target point cloud.
        alpha (float): The transparency level of the points.
        save_path (str, optional): Path to save the plot.
        transpose (bool): Whether to transpose the coordinates of the point clouds.
        **kwargs: Additional keyword arguments for plt.scatter.
    """
    ax = plt.gca()
    
    if transpose:
        ax.scatter(source_pc[:, 1], source_pc[:, 0], s=s, c=source_color, alpha=alpha, label='Source', **kwargs)
        ax.scatter(target_pc[:, 1], target_pc[:, 0], s=s, c=target_color, alpha=alpha, label='Target', **kwargs)
    else:
        ax.scatter(source_pc[:, 0], source_pc[:, 1], s=s, c=source_color, alpha=alpha, label='Source', **kwargs)
        ax.scatter(target_pc[:, 0], target_pc[:, 1], s=s, c=target_color, alpha=alpha, label='Target', **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path is not None:
        save_plot(save_path, dpi=600)

    # plt.legend()
    plt.close()
    

def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=0.5, adaptive=True):
    '''Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    '''
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def save_plot(path, **kw):
    '''Save the current figure without any white margin.'''
    plt.savefig(path, bbox_inches='tight', pad_inches=0, **kw)


def imshow(image, save_path=None, colorbar=False):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = image.squeeze()
    plt.imshow(image, cmap='jet')
    if colorbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if save_path:
        save_plot(save_path, dpi=600)
    plt.show()
    plt.close()
    

def features_to_RGB(*Fs, masks=None, skip=1):
    # import pdb; pdb.set_trace()
    '''Project a list of d-dimensional feature maps to RGB colors using PCA.'''
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)

    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0)
    flatten = normalize(flatten)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(flatten[::skip])
        flatten = pca.transform(flatten)
    else:
        flatten = pca.fit_transform(flatten)
    flatten = (normalize(flatten) + 1) / 2

    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return np.asarray(Fs_rgb).squeeze()


def features_to_tsne(F, perplexity=30, n_components=3, save_path=None):
    from sklearn.manifold import TSNE
    # import pdb; pdb.set_trace()
    C, H, W = F.shape
    F_flat = F.reshape(C, -1).transpose(1, 0).reshape(-1, C)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    F_tsne = tsne.fit_transform(F_flat)
    
    F_tsne = F_tsne.reshape(H, W, n_components)
    
    if n_components == 1:
        F_tsne = np.repeat(F_tsne, 3, axis=-1)  # If 1 component, repeat to make RGB
    elif n_components == 2:
        F_tsne = np.concatenate([F_tsne, np.zeros((H, W, 1))], axis=-1)  # If 2 components, add a zero channel
    
    F_tsne = (F_tsne - F_tsne.min()) / (F_tsne.max() - F_tsne.min() + 1e-6)  # Normalize to [0, 1]
    if save_path is not None:
        img = (F_tsne * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)
    
    return F_tsne


def likelihood_overlay(
    prob, map_viz=None, p_rgb=0.2, p_alpha=1 / 15, thresh=None, cmap='jet'
):
    prob = prob / prob.max()
    cmap = plt.get_cmap(cmap)
    rgb = cmap(prob**p_rgb)
    alpha = prob[..., None] ** p_alpha
    if thresh is not None:
        alpha[prob <= thresh] = 0
    if map_viz is not None:
        faded = map_viz + (1 - map_viz) * 0.5
        rgb = rgb[..., :3] * alpha + faded * (1 - alpha)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb[..., -1] = alpha.squeeze(-1)
    return rgb


def heatmap2rgb(scores, mask=None, clip_min=0.05, alpha=0.8, cmap='jet'):
    min_, max_ = np.quantile(scores, [clip_min, 1])
    scores = scores.clip(min=min_)
    rgb = plt.get_cmap(cmap)((scores - min_) / (max_ - min_))
    if mask is not None:
        if alpha == 0:
            rgb[mask] = np.nan
        else:
            rgb[..., -1] = 1 - (1 - 1.0 * mask) * (1 - alpha)
    return rgb


def plot_pose(axs, xy, yaw=None, s=1 / 35, c='r', a=1, w=0.015, dot=True, zorder=10):
    if yaw is not None:
        yaw = np.deg2rad(yaw)
        uv = np.array([np.sin(yaw), -np.cos(yaw)])
    xy = np.array(xy) + 0.5
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        if isinstance(ax, int):
            ax = plt.gcf().axes[ax]
        if dot:
            ax.scatter(*xy, c=c, s=70, zorder=zorder, linewidths=0, alpha=a)
        if yaw is not None:
            ax.quiver(
                *xy,
                *uv,
                scale=s,
                scale_units='xy',
                angles='xy',
                color=c,
                zorder=zorder,
                alpha=a,
                width=w,
            )


def plot_dense_rotations(
    ax, prob, thresh=0.01, skip=10, s=1 / 15, k=3, c='k', w=None, **kwargs
):
    t = torch.argmax(prob, -1)
    yaws = t.numpy() / prob.shape[-1] * 360
    prob = prob.max(-1).values / prob.max()
    mask = prob > thresh
    masked = prob.masked_fill(~mask, 0)
    max_ = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    mask = (max_[0, 0] == masked.float()) & mask
    indices = np.where(mask.numpy() > 0)
    plot_pose(
        ax,
        indices[::-1],
        yaws[indices],
        s=s,
        c=c,
        dot=False,
        zorder=0.1,
        w=w,
        **kwargs,
    )


def copy_image(im, ax):
    prop = im.properties()
    prop.pop('children')
    prop.pop('size')
    prop.pop('tightbbox', None)
    prop.pop('transformed_clip_path_and_affine')
    prop.pop('window_extent')
    prop.pop('figure')
    prop.pop('transform')
    return ax.imshow(im.get_array(), **prop)


def add_circle_inset(
    ax,
    center,
    corner=None,
    radius_px=10,
    inset_size=0.4,
    inset_offset=0.005,
    color='red',
):
    data_t_axes = ax.transAxes + ax.transData.inverted()
    if corner is None:
        center_axes = np.array(data_t_axes.inverted().transform(center))
        corner = 1 - np.round(center_axes).astype(int)
    corner = np.array(corner)
    bottom_left = corner * (1 - inset_size - inset_offset) + (1 - corner) * inset_offset
    axins = ax.inset_axes([*bottom_left, inset_size, inset_size])
    if ax.yaxis_inverted():
        axins.invert_yaxis()
    axins.set_axis_off()

    c = mpl.patches.Circle(center, radius_px, fill=False, color=color)
    ax.add_patch(copy.deepcopy(c))
    axins.add_patch(c)

    radius_inset = radius_px + 1
    axins.set_xlim([center[0] - radius_inset, center[0] + radius_inset])
    ylim = center[1] - radius_inset, center[1] + radius_inset
    if axins.yaxis_inverted():
        ylim = ylim[::-1]
    axins.set_ylim(ylim)

    for im in ax.images:
        im2 = copy_image(im, axins)
        im2.set_clip_path(c)
    return axins


def plot_bev(bev, uv, yaw, ax=None, zorder=10, **kwargs):
    if ax is None:
        ax = plt.gca()
    h, w = bev.shape[:2]
    tfm = mpl.transforms.Affine2D().translate(-w / 2, -h)
    tfm = tfm.rotate_deg(yaw).translate(*uv + 0.5)
    tfm += plt.gca().transData
    ax.imshow(bev, transform=tfm, zorder=zorder, **kwargs)
    ax.plot(
        [0, w - 1, w / 2, 0],
        [0, 0, h - 0.5, 0],
        transform=tfm,
        c='k',
        lw=1,
        zorder=zorder + 1,
    )


def plot_pose_3dof(ax, xy, yaw, s=1 / 35, c='r', a=1, w=0.015, dot=True, zorder=10):
    uv = np.array([np.sin(yaw), -np.cos(yaw)])
    xy = np.array(xy) + 0.5
    if dot:
        ax.scatter(*xy, c=c, s=70, zorder=zorder, linewidths=0, alpha=a)
    ax.quiver(
        *xy,
        *uv,
        scale=s,
        scale_units='xy',
        angles='xy',
        color=c,
        zorder=zorder,
        alpha=a,
        width=w,
    )


def plot_heatmap(corr, gt_xy_grid, gt_yaw, est_xy_grid, est_yaw, save_path, sensor='vision'):
    fig, ax = plt.subplots()
    if isinstance(corr, torch.Tensor):
        corr = corr.detach().cpu().numpy()
    corr = corr.squeeze()
    if est_xy_grid is None:
        est_xy_grid = np.unravel_index(np.argmax(corr), corr.shape)
    # Convert the image coordinates to the xy coordinates
    gt_xy_grid = np.array([gt_xy_grid[1], gt_xy_grid[0]])
    est_xy_grid = np.array([est_xy_grid[1], est_xy_grid[0]])
    if sensor == 'lidar':
        corr = corr.transpose()
        gt_xy_grid = gt_xy_grid[::-1]
        est_xy_grid = est_xy_grid[::-1]
        gt_yaw = -gt_yaw
        est_yaw = -est_yaw
    plt.imshow(corr, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    plot_pose_3dof(ax, gt_xy_grid, gt_yaw, c='r', s=1 / 35)
    plot_pose_3dof(ax, est_xy_grid, est_yaw, c='k', s=1 / 25)
    axins = add_circle_inset(ax, est_xy_grid)
    axins.scatter(gt_xy_grid[0], gt_xy_grid[1], lw=1, c='r', ec='k', s=50, zorder=15)
    save_plot(save_path, dpi=600)
    plt.close()


def plot_trajectory(map_xys, query_xys, map_idx, query_idx, loc_flag, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(map_xys[:, 0], map_xys[:, 1], label='Map Trajectory', color='gray')
    plt.plot(query_xys[:(query_idx+1), 0], query_xys[:(query_idx+1), 1], label='Query Trajectory', color='blue')
    plt.scatter(map_xys[map_idx, 0], map_xys[map_idx, 1], color='gray', marker='d', alpha=0.5, label='Query Point')
    plt.scatter(query_xys[query_idx, 0], query_xys[query_idx, 1], color='blue', marker='o', alpha=0.5, label='Retrieved Map Point')
    line_color = 'green' if loc_flag else 'red'
    plt.plot([query_xys[query_idx, 0], map_xys[map_idx, 0]], [query_xys[query_idx, 1], map_xys[map_idx, 1]], color=line_color, alpha=0.5)
    plt.xlim([-400, 100])
    plt.ylim([-800, 100])
    plt.xlabel(r'X [m]')
    plt.ylabel(r'Y [m]')
    # plt.title('Trajectory')
    plt.legend(loc='upper left', borderaxespad=0, frameon=False)
    if save_path:
        save_plot(save_path, dpi=600)
    plt.close()


def create_blank_image(H, W, text, save_path=None):
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=600)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    
    ax.text(W / 2, H / 2, text, fontsize=H / 12, fontweight='bold',
            color='red', ha='center', va='center')
    
    if save_path:
        save_plot(save_path, dpi=600)
    plt.close()
    

if __name__=='__main__':
    bev = np.random.random((10, 128, 128))
    bev_tsne = features_to_tsne(bev, save_path='bev_tsne.jpg')

    bev_rgb = features_to_RGB(bev)
    bev_rgb = (bev_rgb * 255).astype(np.uint8)
    
    if bev_rgb.shape[-1] == 4:
        mask = bev_rgb[..., 3]
        bev_rgb = bev_rgb[..., :3]

    output_path = 'bev_rgb.png'
    bev_rgb = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)
    imshow(bev_rgb, output_path)

    # cv2.imwrite(output_path, bev_rgb)