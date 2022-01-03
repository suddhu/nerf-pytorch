import os
from os import environ, path as osp
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def positionquat2tf(position_quat):
    position_quat = np.atleast_2d(position_quat)
    # position_quat : N x 7
    N = position_quat.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
    T[:, 0:3,3] = position_quat[:, :3]
    T[:, 3, 3] = 1
    return T

def loadImagesAndPoses(data_root, split, resize = True, skip = 1):
    """Loads images and corresponding poses for a given model dataset"""
    data_folder = osp.join(data_root, split, "images")
    imageFiles = sorted(os.listdir(data_folder), key=lambda y: int(y.split(".")[0]))
    images = [cv2.imread(osp.join(data_folder, file))  for file in imageFiles[0::skip]]
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  for im in images]

    data_folder = osp.join(data_root, split, "masks")
    maskFiles = sorted(os.listdir(data_folder), key=lambda y: int(y.split(".")[0]))
    masks = [cv2.imread(osp.join(data_folder, file), cv2.IMREAD_GRAYSCALE)  for file in maskFiles[0::skip]]

    # crop data 
    if resize:
        center, h, w = [x//2 for x in images[0].shape[:2]], 400, 400
        images = [im[int(center[0] - h/2):int(center[0] - h/2 + h), int(center[1] - w/2):int(center[1] - w/2 + w)]  for im in images]
        masks = [mask[int(center[0] - h/2):int(center[0] - h/2 + h), int(center[1] - w/2):int(center[1] - w/2 + w)]  for mask in masks]

    poses_path = osp.join(data_root, split, "poses.npz")
    data = np.load(poses_path)
    gelposes = (data['arr_0'])[0::skip, :]
    gelposes = positionquat2tf(gelposes)
    camposes = (data['arr_1'])[0::skip, :]
    camposes = positionquat2tf(camposes)
    print("Loaded {n} images and poses from: {p}".format(n = len(images), p = data_folder))
    return images, masks, camposes, gelposes

def load_nerf_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']

    all_imgs, all_masks, all_camposes, all_gelposes = [], [], [], []
    counts = [0]
    for s in splits:
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        imgs, masks, camposes, gelposes = loadImagesAndPoses(basedir, s, resize=True, skip=skip)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        masks = (np.array(masks) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        camposes = np.array(camposes).astype(np.float32)
        gelposes = np.array(gelposes).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_masks.append(masks)
        all_camposes.append(camposes)
        all_gelposes.append(gelposes)

    # train, val, test splits 
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    masks = np.concatenate(all_masks, 0)
    poses = np.concatenate(all_camposes, 0)
    
    H, W = imgs[0].shape[:2]
    focal = np.load(osp.join(basedir, "f.npy"))
    
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_poses = poses

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        masks_half_res = np.zeros((masks.shape[0], H, W))

        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        
        for i, mask in enumerate(masks):
            masks_half_res[i] = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
        
        imgs = imgs_half_res
        masks = masks_half_res
        
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    return imgs, masks, poses, render_poses, [H, W, focal], i_split


