""" Helper methods for loading and parsing KITTI data.

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import os
import glob


import math


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):

        calibs = self.read_calib_file(calib_filepath)

        # Depth_Camera
        self.depth_camera = {
            "intrinsics": self.set_camera_intrinsics(calibs['depth_intrinsics']),
            "extrinsics": self.set_camera_extrinsics(calibs['depth_extrinsics'])
        }

        # Rgb_Camera
        self.rgb_camera = {
            "intrinsics": self.set_camera_intrinsics(calibs['rgb_intrinsics']),
            "extrinsics": self.set_camera_extrinsics(calibs['rgb_extrinsics'])
        }

        # self.print_camera_parameters(self.depth_camera)

    def print_camera_parameters(self, camera):
        print("camera_matrix :",
              np.around(camera["intrinsics"]['camera_matrix'], 2),
              "\n")
        print("radial_distortion :",
              np.around(camera["intrinsics"]['radial_distortion'], 2),
              "\n")
        print("tangential_distortion :",
              np.around(camera["intrinsics"]['tangential_distortion'], 2),
              "\n")
        print("center_distortion :",
              np.around(camera["intrinsics"]['center_distortion'], 2),
              "\n")
        print("radius_distortion :",
              np.around(camera["intrinsics"]['radius_distortion'], 2),
              "\n")
        print("R :",
              np.around(camera["extrinsics"]['R'], 2),
              "\n")
        print("T :",
              np.around(camera["extrinsics"]['T'], 2),
              "\n")

    def set_camera_intrinsics(self, param):
        _dict = {
            "H": param[0], "W": param[1],
            "camera_matrix": param[2:6],
            "radial_distortion": param[6:12],
            "tangential_distortion": param[12:14],
            "center_distortion": param[14:16],
            "radius_distortion": param[16],
        }
        return _dict

    def set_camera_extrinsics(self, param):
        matrix = np.reshape(param, [3, 4])

        _dict = {
            "R": matrix[:, 0:3],
            "T": np.reshape(matrix[:, 3], (3, 1)),
        }
        return _dict

    def read_calib_file(self, path):
        data = {}

        print(path)
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(' ', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def get_uv_depth(self, depth):
        rows = depth.shape[0]
        cols = depth.shape[1]

        print("rows", rows)
        print("cols", cols)

        u_ = np.arange(rows, dtype=int)
        v_ = np.arange(cols, dtype=int)

        u = np.reshape(u_, (-1, 1)) * np.ones(cols)
        v = np.reshape(np.ones(rows), (-1, 1)) * v_

        uv_depth = np.reshape(v, (-1, 1))
        uv_depth = np.append(uv_depth, np.reshape(u, (-1, 1)), axis=1)
        uv_depth = np.append(uv_depth, np.reshape(depth, (-1, 1)), axis=1)

        uv_depth.astype(float)

        np.savetxt("uv_depth.txt", uv_depth, newline="\n", fmt='%1.1f')


        return uv_depth

    def get_distortion_coef(self, camera):
        radial_dist = camera["intrinsics"]['radial_distortion']
        tangential_dist = camera["intrinsics"]['tangential_distortion']

        k1 = radial_dist[0]
        k2 = radial_dist[1]
        k3 = radial_dist[2]
        k4 = radial_dist[3]
        k5 = radial_dist[4]
        k6 = radial_dist[5]

        p1 = tangential_dist[0]
        p2 = tangential_dist[1]

        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        dist_coeffs = dist_coeffs.reshape(-1, 1)
        dist_coeffs = np.array(dist_coeffs)
        dist_coeffs.astype(float)
        return dist_coeffs

    def get_rgb_distortion(self):
        return self.get_distortion_coef(self.rgb_camera)

    def get_depth_distortion(self):
        return self.get_distortion_coef(self.depth_camera)

    def get_camera_matrix(self, camera):
        _temp = camera["intrinsics"]['camera_matrix']
        k = np.zeros((3, 3))
        k[0, 0] = _temp[0]
        k[1, 1] = _temp[1]
        k[0, 2] = _temp[2]
        k[1, 2] = _temp[3]
        k[2, 2] = 1

        return k

    def get_rgb_camera_matrix(self):
        return self.get_camera_matrix(self.rgb_camera)

    def get_depth_camera_matrix(self):
        return self.get_camera_matrix(self.depth_camera)

    def get_size(self, camera):
        H = camera["intrinsics"]['H']
        W = camera["intrinsics"]['W']
        return int(W), int(H)

    def get_rgb_size(self):
        return self.get_size(self.rgb_camera)

    def get_depth_size(self):
        return self.get_size(self.depth_camera)

    def get_extrinsics(self, camera):
        R = camera["extrinsics"]['R']
        T = camera["extrinsics"]['T']
        return R, T

    def get_rgb_extrinsics(self):
        return self.get_extrinsics(self.rgb_camera)

    def get_depth_extrinsics(self):
        return self.get_extrinsics(self.depth_camera)

'''
        # XYZ to rgb
        #R = self.rgb_camera["extrinsics"]['R']
        #T = self.rgb_camera["extrinsics"]['T']

        y = np.multiply((1 / fy), (uv_depth[:, 2]))

        xyz_rgb = R.dot(XYZ.transpose())
        xyz_rgb = xyz_rgb + T
        xyz_rgb = xyz_rgb.transpose()

        # to camera
        camera_matrix = self.rgb_camera["intrinsics"]['camera_matrix']
        radial_dist = self.rgb_camera["intrinsics"]['radial_distortion']
        tangential_dist = self.rgb_camera["intrinsics"]['tangential_distortion']



        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        k1 = radial_dist[0]
        k2 = radial_dist[1]
        k3 = radial_dist[2]
        k4 = radial_dist[3]
        k5 = radial_dist[4]
        k6 = radial_dist[5]

        p1 = tangential_dist[0]
        p2 = tangential_dist[1]

        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

        z = xyz_rgb[:, 2]
        xy = xyz_rgb[:, 0:2]
        print(z.shape)

        xy[:, 0] = np.multiply(xy[:, 0], np.reciprocal(z))
        xy[:, 1] = np.multiply(xy[:, 1], np.reciprocal(z))

        uv_rgb = np.zeros_like(xy)

        uv_rgb[:, 0] = np.multiply(xy[:, 0], fx) + cx
        uv_rgb[:, 1] = np.multiply(xy[:, 0], fy) + cy

        ## apply distortion first

        #uv_rgb = uv_rgb.astype(int)
        #np.savetxt("uv_rgb.txt", uv_rgb, newline="\n", fmt='%1.3f')


        #H = self.rgb_camera["intrinsics"]['H']
        #W = self.rgb_camera["intrinsics"]['W']

        #uv_rgb_mask = np.zeros_like(uv_rgb)
        #uv_rgb_mask[:, 0] = uv_rgb[:, 0] < H

        #uv_rgb_mask[:, 1] = uv_rgb[:, 1] < W

        #uv_rgb = uv_rgb[uv_rgb_mask]
        #uv_rgb = np.dot(camera_matrix_rgb, xyz_rgb)



        np.savetxt("uv_rgb_mask.txt", uv_rgb_mask, newline="\n", fmt='%1.3f')

'''


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def load_bbox_2d(path):
    detections = [x for x in glob.glob(path + '/*')]
    return detections


def load_image(path, disp=False):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if disp:
        plt.imshow(image)
        plt.show()
    return image


def load_depth(path,  disp=False):
    _depth, _scale = readPFM(path)
    _depth = _depth*1000  #convert back to milimiters
    if disp:
        plt.imshow(_depth, cmap='jet')
        plt.show()
    return _depth


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0];
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1];
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2];
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''

    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = project_to_image(np.transpose(orientation_3d), P);
    return orientation_2d, np.transpose(orientation_3d)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image


def readPFM(path):

    file = open(path, 'rb')

    bands = file.readline().decode('utf-8').rstrip()

    if bands == 'PF':
        color = True
    elif bands == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    width = int(file.readline())
    height = int(file.readline())
    scale = float(file.readline())

    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')

    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



'''
Save a Numpy array to a PFM file.
'''


def save_pfm(file, image, scale=1):
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)
