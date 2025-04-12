import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def q2r(q):
    # Rotation matrix from Hamiltonian quaternion
    # Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    w, x, y, z = tuple(q)

    n = 1.0/np.sqrt(x*x+y*y+z*z+w*w)
    x *= n
    y *= n
    z *= n
    w *= n
    r = np.array([[1.0 - 2.0*y*y - 2.0*z*z, 2.0*x*y - 2.0*z*w, 2.0*x*z + 2.0*y*w],
                  [2.0*x*y + 2.0*z*w, 1.0 - 2.0*x*x - 2.0*z*z, 2.0*y*z - 2.0*x*w],
                  [2.0*x*z - 2.0*y*w, 2.0*y*z + 2.0*x*w, 1.0 - 2.0*x*x - 2.0*y*y]])
    return r


def m2xyz_q(m):
    assert m.shape[-2:] == (4, 4)
    t = m[:3, 3]
    q = R.from_matrix(m[:3, :3]).as_quat()
    
    return t, q


def m2ypr(m):
    # Get yaw, pitch, roll angles from 4x4 transformation matrix
    # Based on formulas in Section 2.5.1 in:
    # A tutorial on SE(3) transformation parameterizations and on-manifold optimization
    # https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    assert m.shape[-2:] == (4, 4)
    if m.ndim == 2:
        pitch = np.arctan2(-m[2][0], np.sqrt(m[0][0]**2 + m[1][0]**2))
        # We do not handle degenerate case, when pitch is 90 degrees a.k.a. gimball lock
        assert not np.isclose(np.abs(pitch), np.pi/2)
        yaw = np.arctan2(m[1][0], m[0][0])
        roll = np.arctan2(m[2][1], m[2][2])
    else:
        yaw = np.zeros(m.shape[0])
        pitch = np.zeros(m.shape[0])
        roll = np.zeros(m.shape[0])
        for i, mi in enumerate(m):
            pitch_i = np.arctan2(-mi[2][0], np.sqrt(mi[0][0]**2 + mi[1][0]**2))
            # We do not handle degenerate case, when pitch is 90 degrees a.k.a. gimball lock
            assert not np.isclose(np.abs(pitch_i), np.pi/2)
            yaw_i = np.arctan2(mi[1][0], mi[0][0])
            roll_i = np.arctan2(mi[2][1], mi[2][2])    
            yaw[i] = yaw_i    
            pitch[i] = pitch_i    
            roll[i] = roll_i    

    return yaw, pitch, roll


def m2xyz_ypr(m):
    # Get yaw, pitch, roll angles from 4x4 transformation matrix
    # Based on formulas in Section 2.5.1 in:
    # A tutorial on SE(3) transformation parameterizations and on-manifold optimization
    # https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    assert m.shape[-2:] == (4, 4)
    yaw, pitch, roll = m2ypr(m)
    return m[..., 0, 3], m[..., 1, 3], m[..., 2, 3], yaw, pitch, roll


def ypr2m(yaw, pitch, roll):
    # Construct 4x4 transformation matrix with rotation part set to given yaw, pitch, roll. Translation is set to 0.
    # Based on formulas in Section 2.2.1
    m = np.array([[np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
                   np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll), 0.],
                  [np.sin(yaw) * np.cos(pitch), np.sin(roll) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
                   np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll), 0.],
                  [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll), 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)

    return m


def xyz_ypr2m(x, y, z, yaw, pitch, roll):
    # Construct 4x4 transformation matrix with given translation and rotation part set to given yaw, pitch, roll.
    # Based on formulas in Section 2.2.1
    m = ypr2m(yaw, pitch, roll)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    # Apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
    assert pc.ndim == 2
    # n_dim = pc.shape[1]
    n_dim = 3
    # assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc[:,:3] = pc[:,:3] @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]
    return pc


def relative_pose(m1, m2):
    # !!! DO NOT USE THIS FUNCTION FOR MULRAN POSES !!!
    # SE(3) pose is 4x 4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: coords in camera/lidar1 reference frame -> coords in camera/lidar2 reference frame
    #          relative pose of the first camera with respect to the second camera
    return np.linalg.inv(m2) @ m1


def relative_pose_batch(m1, m2):
    """
    Calculate the relative pose between two sets of poses represented as transformation matrices.

    Args:
        m1 (torch.Tensor): Tensor of shape (n, 4, 4) representing the first set of poses.
        m2 (torch.Tensor): Tensor of shape (n, 4, 4) representing the second set of poses.

    """
    if isinstance(m1, np.ndarray):
        m1 = torch.from_numpy(m1).float()
    if isinstance(m2, np.ndarray):
        m2 = torch.from_numpy(m2).float()
    m = torch.matmul(m2.inverse(), m1)

    return m