''' use 2D detections to create Frustums

Author: Tarek BOUAMER
Institute: ICG TU graz
Date: Feb 2020
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2

import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches

from PIL import Image, ImageDraw

import utils as util

import _pickle as pickle

#from kitti_object import *
import argparse


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def demo():
    import mayavi.mlab as mlab

    from mayavi.viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
    calib = dataset.get_calibration(data_idx)

    ## Draw lidar in rect camera coord
    # print(' -------- LiDAR points in rect camera coordination --------')
    # pc_rect = calib.project_velo_to_rect(pc_velo)
    # fig = draw_lidar_simple(pc_rect)
    # raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # show_lidar_with_boxes(pc_velo, objects, calib)
    # raw_input()
    show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    raw_input()

    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:, 0:2] = imgfov_pts_2d
    cameraUVDepth[:, 2] = imgfov_pc_rect[:, 2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin, ymin, xmax, ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])



def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare': continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


class Scene(object):
    '''Load and parse Scene data '''

    def __init__(self, root_dir):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir

        #TODO: further modification possible
        self.image_dir = os.path.join(self.root_dir, 'image')
        self.depth_dir = os.path.join(self.root_dir, 'depth')

        self.calib_dir = os.path.join(self.root_dir, 'calib')

        self.bbox_2d_dir = os.path.join(self.root_dir, 'bbox_2d')

        self.bbox_3d_dir = os.path.join(self.root_dir, 'bbox_3d')

        self.query = self.get_query_list()
        self.num_samples = len(self.query)

        self.calib = self.get_calibration()

    def __len__(self):
        return self.num_samples

    def get_query_list(self):
        __list = []
        for name in glob.glob(self.image_dir + '/*'):
            Name, ext = os.path.basename(name).split('.')
            __list.append(Name)
        __list.sort()
        return __list

    def get_rgb_image(self, idx):
        image_path = os.path.join(self.image_dir, str(idx) + '.png')
        return util.load_image(image_path, False)

    def get_depth_image(self, idx):
        depth_path = os.path.join(self.depth_dir, str(idx) + '.pfm')
        return util.load_depth(depth_path, False)

    def get_calibration(self):
        calib_path = os.path.join(self.calib_dir, 'calibration.txt')
        return util.Calibration(calib_path)

    def get_bbox_2d(self, idx):
        bbox_2d_path = os.path.join(self.bbox_2d_dir, idx)
        return sorted(util.load_bbox_2d(bbox_2d_path))

    def extract_detection(self, path):
        det = np.load(path, allow_pickle=True)
        id = det[()]['id']
        bbox = det[()]['bbox']
        cls_pred = det[()]['cls_pred']

        obj_pred = det[()]['obj_pred']
        msk_pred = det[()]['msk_pred']

        bbox = [(bbox['y1']),
                (bbox['x1']),
                (bbox['y2']),
                (bbox['x2']),
                ]
        return id, bbox, cls_pred, obj_pred, msk_pred

    def get_undistorted_image(self, idx):
        image = self.get_rgb_image(idx)

        dist_coeffs = self.calib.get_rgb_distortion()
        camera_matrix = self.calib.get_rgb_camera_matrix()

        W, H = self.calib.get_rgb_size()
        size = (W, H)

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, 0)
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, size, 5)

        self.new_rgb_camera_matrix = new_camera_matrix
        self.mapx = mapx
        self.mapy = mapy

        image_unistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        return image_unistorted, new_camera_matrix, size

    def get_undistorted_depth(self, idx):
        depth = self.get_depth_image(idx)

        dist_coeffs = self.calib.get_depth_distortion()
        camera_matrix = self.calib.get_depth_camera_matrix()

        W, H = self.calib.get_rgb_size()
        size = (W, H)

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, 0)

        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, size, 5)

        depth_unistorted = cv2.remap(depth, mapx, mapy, cv2.INTER_LINEAR)

        self.new_depth_camera_matrix = new_camera_matrix

        return depth_unistorted, new_camera_matrix, size

    def get_uv_depth(self, depth):

        rows = depth.shape[0]
        cols = depth.shape[1]

        u_ = np.arange(cols, dtype=int)
        v_ = np.arange(rows, dtype=int)

        u = np.reshape(np.ones(rows), (-1, 1)) * u_
        v = np.reshape(v_, (-1, 1)) * np.ones(cols)

        uv_depth = np.c_[np.reshape(u, (-1, 1)), np.reshape(v, (-1, 1)), np.reshape(depth, (-1, 1))]

        np.savetxt("uv_depth.txt", uv_depth, newline="\n", fmt='%1.1f')


        return uv_depth

    def depthTo3Dpoints(self, depth, camera_matrix):

        uv_depth = self.get_uv_depth(depth)

        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]

        x = np.multiply((1 / fx), uv_depth[:, 2])
        x = np.multiply(x, np.subtract(uv_depth[:, 0], cx))

        y = np.multiply((1 / fy), (uv_depth[:, 2]))
        y = np.multiply(y, np.subtract(uv_depth[:, 1], cy))

        z = uv_depth[:, 2]

        XYZ = np.c_[x, y, z]

        return XYZ

    def projetc3DpointsToImage(self, points3D, camera_matrix):

        dist = self.calib.get_rgb_distortion()

        R , T = self.calib.get_rgb_extrinsics()
        R_r = cv2.Rodrigues(R)[0]

        size = self.calib.get_rgb_size()
        W, H = size

        # get 3d points
        # add distorion here

        pts2d, _ = cv2.projectPoints(points3D, R_r, T, camera_matrix, 0)
        pts2d = np.squeeze(pts2d)

        x = pts2d[:, 0]
        y = pts2d[:, 1]

        maskx = (x > 0)*1
        maskx *= (x < W)*1
        masky = (y > 0)*1
        masky *= (y < H)*1

        mask = np.logical_and(maskx, masky)
        mask = np.c_[mask, mask]

        pts2d = np.multiply(pts2d, mask)
        pts2d = np.round(pts2d, decimals=0)

        pts2d = pts2d[~np.isnan(pts2d).any(axis=1)]
        pts2d = pts2d[~np.all(pts2d == 0, axis=1)]

        pts2d = np.asarray((pts2d))
        print(pts2d.shape)


        return pts2d

    def undistoredImagePoints(self, points, new_camera_matrix):

        points = points[:, 0:2].astype('float32')
        points = np.expand_dims(points, axis=1)  # (n, 1, 2)

        dist_coeffs = self.calib.get_rgb_distortion()
        camera_matrix = self.calib.get_rgb_camera_matrix()

        fx = new_camera_matrix[0, 0]
        fy = new_camera_matrix[1, 1]
        cx = new_camera_matrix[0, 2]
        cy = new_camera_matrix[1, 2]

        points = np.squeeze(cv2.undistortPoints(points, camera_matrix, dist_coeffs))

        points_unistorted = np.empty_like(points)
        for i, (px, py) in enumerate(points):
            points_unistorted[i, 0] = px * fx + cx
            points_unistorted[i, 1] = py * fy + cy

        return points_unistorted

    def undistoredImage(self, image, new_camera_matrix):
        dist_coeffs = self.calib.get_rgb_distortion()
        camera_matrix = self.calib.get_rgb_camera_matrix()

        image_unistorted = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        return image_unistorted

    def imageT3Dpoints(self, uv_depth):

        fx = self.new_rgb_camera_matrix[0][0]
        fy = self.new_rgb_camera_matrix[1][1]
        cx = self.new_rgb_camera_matrix[0][2]
        cy = self.new_rgb_camera_matrix[1][2]

        x = np.multiply((1 / fx), uv_depth[:, 2])
        x = np.multiply(x, np.subtract(uv_depth[:, 0], cx))

        y = np.multiply((1 / fy), (uv_depth[:, 2]))
        y = np.multiply(y, np.subtract(uv_depth[:, 1], cy))

        z = uv_depth[:, 2]

        XYZ = np.c_[x, y, z]
        return XYZ

def extract_frustum_data_rgb_detection(scene_path, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       point_cloud_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
    '''
    sc = Scene(scene_path)   ## init scene
    print(sc.__len__(), " Images have been founed ...")

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    for det_idx in sc.query:
        print("image (  ", det_idx, "  )")
        undis_image, k_rgb, image_size = sc.get_undistorted_image(det_idx)
        undis_depth, k_depth, depth_size = sc.get_undistorted_depth(det_idx)

        points3D = sc.depthTo3Dpoints(undis_depth, k_depth)
        points2D = sc.projetc3DpointsToImage(points3D, k_rgb)

        detections = sc.get_bbox_2d(det_idx)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(25, 10)
        ax.imshow(undis_image)
        #plt.scatter(points2D[:, 0], points2D[:, 1], 1, color='red')

        for det in detections:

            _id, _bbox, _cls_pred, _obj_pred, _msk_pred = sc.extract_detection(str(det))

            _msk_pred = sc.undistoredImage(_msk_pred, k_rgb)

            print("Mask   ", _msk_pred.shape)

            _bbox_np = np.reshape(_bbox, (-1, 2))

            bbox = sc.undistoredImagePoints(_bbox_np, k_rgb)
            x1 = int(bbox[0, 0])
            y1 = int(bbox[0, 1])
            x2 = int(bbox[1, 0])
            y2 = int(bbox[1, 1])

            if y2-y1 <= img_height_threshold:
                continue

            # box stuff
            bbox_image = undis_image[y1:y2, x1:x2]

            x = points2D[:, 0]
            y = points2D[:, 1]
            maskx = (x >= x1) * 1
            maskx *= (x < x2) * 1

            masky = (y > y1) * 1
            masky *= (y < y2) * 1

            mask = np.logical_and(maskx, masky)
            mask = np.c_[mask, mask]
            _box_points2D = np.multiply(points2D, mask)
            _box_points2D = _box_points2D[~np.all(_box_points2D == 0, axis=1)]

            np.savetxt("box_points2D.txt", _box_points2D, newline="\n", fmt='%1.1f')

            box_points2D = np.zeros_like(_box_points2D)
            for i, point in enumerate(_box_points2D):

                px = int(point[0])
                py = int(point[1])
                print(i,    px,     py)
                box_points2D[i, :] = point * int(_msk_pred[py, px])

            box_points2D = box_points2D[~np.all(box_points2D == 0, axis=1)]

            if len(box_points2D) <= point_cloud_threshold:
                continue

            # Get frustum angle (according to center pixel in 2D BOX)
            bbox_center = [x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]
            x0, y0 = bbox_center
            z0 = 20
            bbox_center_uv_depth = np.array([[x0, y0, z0]])

            bbox_center_3D = sc.imageT3Dpoints(bbox_center_uv_depth)
            frustum_angle = np.arcsin(bbox_center_3D[0, 0]/bbox_center_3D[0, 2])
            print("bbox_center ", bbox_center)
            print("frustum point ", bbox_center_3D)
            print("frustum_angle ", frustum_angle*180/np.pi)

            # Vis
            plt.imshow(undis_image)
            plt.scatter(box_points2D[:, 0], box_points2D[:, 1], 1, color='green')
            plt.scatter(bbox[:, 0], bbox[:, 1], 5, color='red')

            plt.scatter(x0, y0, 3, color='blue')
            rect = patches.Rectangle((x1, y1), x2 - x1 + 1, y2 - y1 + 1,  linewidth=0.5, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        plt.show()

        id_list.append(det_idx+'.png')
        type_list.append(_cls_pred)
        box2d_list.append(bbox)
        prob_list.append(_obj_pred)
        #input_list.append(pc_in_box_fov)   ### add the filtred points only :) proceed with detector 
        frustum_angle_list.append(frustum_angle)


def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format.

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')

    args = parser.parse_args()

    # set path to working dir
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # set path to dataset dir
    ROOT_DIR = '/media/tarek/c0f263ed-e006-443e-8f2a-5860fecd27b5/frustum-pointnets/azure/'

    sys.path.append(BASE_DIR)
    print(BASE_DIR)
    print(ROOT_DIR)

    if args.demo:
        demo()
        exit()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_azure_'


    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection( \
            ROOT_DIR,       # TODO: support of many scenes ?
            os.path.join(ROOT_DIR, output_prefix + '3D.pickle'),
            viz=True,
            type_whitelist=type_whitelist)
