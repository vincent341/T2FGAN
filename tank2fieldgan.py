
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import cv2
import matplotlib.pyplot as plt
from numpy import matlib
import shapely
from shapely.geometry import Polygon
from utils import dataset_util
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--fpositionlip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
parser.add_argument("--heading", type=float, default=0.0, help="heading")
parser.add_argument("--pitch", type=float, default=0.0, help="pitch")
parser.add_argument("--roll", type=float, default=0.0, help="roll")

parser.add_argument("--Tx", type=float, default=0.0, help="x of Translation Matrix,horisontal")
parser.add_argument("--Ty", type=float, default=0.0, help="y of Translation Matrix,vertical")
parser.add_argument("--Tz", type=float, default=0.0, help="z of Translation Matrix,Depth")

parser.add_argument("--lx", type=float, default=0.0, help="x position of light source")
parser.add_argument("--ly", type=float, default=0.0, help="y position of light source")
parser.add_argument("--lz", type=float, default=0.0, help="z position of light source")
parser.add_argument("--betar", type=float, default=0.02, help="beta of red")
parser.add_argument("--betag", type=float, default=0.02, help="beta of green")
parser.add_argument("--betab", type=float, default=0.02, help="beta of blue")
parser.add_argument("--light", type=float, default=50, help="light intensity")
parser.add_argument("--g", type=float, default=0.6, help="parameter of  phase function")
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train,gen_layers_outputs")

class Camera(object):
    def __init__(self,imgwidth,imgheight,u0,v0,kx,ky):
        self.imgwidth = int(imgwidth)
        self.imgheight = int(imgheight)
        self.u0 = u0
        self.v0 = v0
        self.kx = kx
        self.ky = ky
    def myprint(self):
        print('imgwidth,',self.imgwidth,'\nimgheight',self.imgheight,
              '\nu0,',self.u0,'\nv0,',self.v0,'\nkx,',self.kx,'\nky,',self.ky)

class bbox(object):
    def __init__(self,lefttoprow,lefttopcol,width,height):
        self.lefttoprow = lefttoprow
        self.lefttopcol = lefttopcol
        self.width = width
        self.height =height

def my_preprocess_np(image):
    return image * 2 - 1

def Generate_bak():
    def generate(cam,param):
        imageheight = cam.imgheight
        imagewidth = cam.imgwidth

        z=param['z']
        col_ind_mat_pl=param['col_ind_mat_pl']
        row_ind_mat_pl=param['row_ind_mat_pl']
        u0_pl = param['u0_pl']
        v0_pl = param['v0_pl']
        kx_pl = param['kx_pl']
        ky_pl = param['ky_pl']


        light_x_v = param['light_x_v']
        light_y_v = param['light_y_v']
        light_z_v = param['light_z_v']
        belta_R_v = param['belta_R_v']
        belta_G_v = param['belta_G_v']
        belta_B_v = param['belta_B_v']
        light_v = param['light_v']
        g_v = param['g_v']

        g_v = tf.minimum(g_v, 1.0)
        g_v = tf.maximum(g_v, -1.0)  # -1<g<1
        belta_R_v = tf.maximum(belta_R_v, 0.0)
        belta_G_v = tf.maximum(belta_G_v, 0.0)
        belta_B_v = tf.maximum(belta_B_v, 0.0)

        x1_mat = (col_ind_mat_pl - u0_pl) * z / kx_pl
        y1_mat = (row_ind_mat_pl - v0_pl) * z / ky_pl
        y1_mat = -y1_mat

        ALdotAO = (x1_mat - light_x_v) * x1_mat + (y1_mat - light_y_v) * y1_mat + (
                z - light_z_v) * z  # A is object,O camera,L is light
        AL_len = tf.sqrt(tf.square(x1_mat - light_x_v) + tf.square(y1_mat - light_y_v) + tf.square(z - light_z_v))
        AO_len = tf.sqrt(tf.square(x1_mat) + tf.square(y1_mat) + tf.square(z))
        cos = -ALdotAO / (AL_len * AO_len + 1e-8)  # cos(pi-a)=-cos(a)
        domtmp = 4 * np.pi * pow((1 + tf.square(g_v) - 2 * g_v * cos), 1.5) + 1e-8
        pg = (1 - tf.square(g_v)) / domtmp
        t1_R = light_v * pg * tf.exp(-belta_R_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_R = tf.reduce_sum(t1_R, axis=0)

        t1_G = light_v * pg * tf.exp(-belta_G_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_G = tf.reduce_sum(t1_G, axis=0)

        t1_B = light_v * pg * tf.exp(-belta_B_v * (AL_len + AO_len)) / (AL_len + 1e-8)
        disp6_B = tf.reduce_sum(t1_B, axis=0)

        BGR_hat = tf.stack([disp6_B, disp6_G, disp6_R], axis=2)
        return  BGR_hat
    rescale_factor = 4.0
    kx = 633.5915 / rescale_factor
    ky = 695.2624 / rescale_factor
    u0 = 364.5163 / rescale_factor
    v0 = 234.3574 / rescale_factor
    imgwid = int(720 // rescale_factor)
    imgheight = int(576 // rescale_factor)

    light_x = a.lx
    light_y = a.ly
    light_z = a.lz
    belta_R = a.betar
    belta_G = a.betag
    belta_B = a.betab
    light = a.light
    g = a.g


    Cam = Camera(imgwid, imgheight, u0, v0, kx, ky)
    Z_refer = 50
    g_bak = tf.Graph()
    with g_bak.as_default():
        z = tf.placeholder(tf.float32, shape=[Z_refer, imgheight, imgwid])  # (Z_refer,height,width)
        col_ind_mat_pl = tf.placeholder(tf.float32, shape=[imgheight, imgwid])
        row_ind_mat_pl = tf.placeholder(tf.float32, shape=[imgheight, imgwid])
        u0_pl = tf.placeholder(tf.float32, shape=[1, 1])
        v0_pl = tf.placeholder(tf.float32, shape=[1, 1])
        kx_pl = tf.placeholder(tf.float32, shape=[1, 1])
        ky_pl = tf.placeholder(tf.float32, shape=[1, 1])

        light_x_v = tf.placeholder(tf.float32, shape=[1, 1])
        light_y_v = tf.placeholder(tf.float32, shape=[1, 1])
        light_z_v = tf.placeholder(tf.float32, shape=[1, 1])
        belta_R_v = tf.placeholder(tf.float32, shape=[1, 1])
        belta_G_v = tf.placeholder(tf.float32, shape=[1, 1])
        belta_B_v = tf.placeholder(tf.float32, shape=[1, 1])
        light_v = tf.placeholder(tf.float32, shape=[1, 1])
        g_v = tf.placeholder(tf.float32, shape=[1, 1])
        param={'z':z,
               'col_ind_mat_pl':col_ind_mat_pl,
               'row_ind_mat_pl':row_ind_mat_pl,
               'u0_pl':u0_pl,
               'v0_pl': v0_pl,
               'kx_pl':kx_pl,
               'ky_pl':ky_pl,
               'light_x_v':light_x_v,
               'light_y_v':light_y_v,
               'light_z_v':light_z_v,
               'belta_R_v':belta_R_v,
               'belta_G_v':belta_G_v,
               'belta_B_v':belta_B_v,
               'light_v':light_v,
               'g_v':g_v}



        BGR_hat= generate(Cam,param)
        intial_op = tf.global_variables_initializer()

    z_tmp = np.arange(Z_refer, dtype=np.float32)
    z_tmp = np.tile(z_tmp, (imgheight, imgwid, 1))
    z_val = np.transpose(z_tmp, (2, 0, 1))
    col_ind = np.arange(imgwid)  # generate 0,1,2...imgwidth
    col_ind_mat_val = np.matlib.repmat(col_ind, imgheight, 1)

    row_ind = np.arange(imgheight)
    row_ind_mat_val = np.transpose(np.matlib.repmat(row_ind, imgwid, 1))


    with tf.Session(graph=g_bak) as sess:
        sess.run(intial_op)
        [result] = sess.run([BGR_hat],feed_dict={col_ind_mat_pl: col_ind_mat_val,
                                             row_ind_mat_pl: row_ind_mat_val,
                                             u0_pl: [[u0]], v0_pl: [[v0]], kx_pl: [[kx]],
                                             ky_pl: [[ky]],z: z_val,light_x_v:[[light_x]],light_y_v:[[light_y]],
                                                 light_z_v:[[light_z]],belta_R_v:[[belta_R]],belta_G_v:[[belta_G]],
                                                 belta_B_v:[[belta_B]],light_v:[[light]],g_v:[[g]]})
        res_norm = cv2.normalize(result, None,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)# to range [0,1]

    return  res_norm #range [0,1]

def my_readimg_np(pathname):
    bak_gen=Generate_bak()# generate background image according to the model
    bak_gen2 = cv2.cvtColor(bak_gen, cv2.COLOR_BGR2RGB) #convert BGR2RGB to keep consistent
    bak_gen=bak_gen2
    img=cv2.imread(pathname)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.float32(img/255.0)
    height = img.shape[0]
    bak_gen = cv2.resize(bak_gen,(height,height))
    target = my_preprocess_np(img[:,:height,:])
    bak = my_preprocess_np(bak_gen)
    eulermat = [a.pitch,a.heading,a.roll]
    Tmatrix = np.array([[a.Tx, a.Ty, a.Tz]], dtype=np.float32).transpose()
    lightmark_gen,bbox = Generatelightmask_frompinhole(eulermat,Tmatrix)

    if bbox==():#indicates invalid postion of docking station in settings
        print('Invalid settings.\n')
        return [],[],bbox
    #resize bbox to a new one
    bbox_re = bbox
    col_min_re=bbox[0]*(height/lightmark_gen.shape[1])
    col_max_re = bbox[2]*(height/lightmark_gen.shape[1])
    row_min_re= bbox[1] * (height / lightmark_gen.shape[0])
    row_max_re = bbox[3] * (height / lightmark_gen.shape[0])
    bbox_re=(col_min_re,row_min_re,col_max_re,row_max_re)
    lightmark_gen = cv2.resize(lightmark_gen, (height, height), interpolation=cv2.INTER_CUBIC)#from 720*576 to 256*256
    ret, lightmark_gen = cv2.threshold(lightmark_gen, 30, 255, cv2.THRESH_BINARY)  # further improve light mask
    lightmark_gen = np.float32(lightmark_gen / 255.0)
    lightmark = my_preprocess_np(lightmark_gen)
    lightmark = np.expand_dims(lightmark,axis=2)
    inputs = np.concatenate([bak,lightmark],axis=2)
    inputs=inputs[:,:,:4]
    inputs= np.expand_dims(inputs,axis=0)
    target = np.expand_dims(target,axis=0)
    return inputs,target,bbox_re

def Generatelightmask_frompinhole(Euler,T):
    #Generating light masks from a pinhole camera model given:
    #Euler angles [a,b,c].a:roll,b:heading,c:
    # T: translation matrix,e.g.T = np.array([[1000,0,10000]],dtype=np.float32).transpose()
    #[->,↓，depth]
    def eulerAnglesToRotationMatrix(theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R
    kx = 633.5915
    ky = 695.2624
    u0 = 364.5163
    v0 = 234.3574
    imgwid = int(720)
    imgheight = int(576)
    Euler_rad = np.deg2rad(Euler)
    R_euler = eulerAnglesToRotationMatrix(Euler_rad)

    r_station = 1027.0 + 34.0  # mm; r of the docking station + r of lights
    deg_1 = 63.0
    deg_2 = 27.0
    deg_3 = 333.0
    deg_4 = 297.0
    deg_5 = 243.0
    deg_6 = 207.0
    deg_7 = 153.0
    deg_8 = 117.0
    x_light1 = r_station * math.cos(np.deg2rad(deg_1))
    y_light1 = -r_station * math.sin(np.deg2rad(deg_1))# originally no -

    x_light2 = r_station * math.cos(np.deg2rad(deg_2))
    y_light2 = -r_station * math.sin(np.deg2rad(deg_2))

    x_light3 = r_station * math.cos(np.deg2rad(deg_3))
    y_light3 = -r_station * math.sin(np.deg2rad(deg_3))

    x_light4 = r_station * math.cos(np.deg2rad(deg_4))
    y_light4 = -r_station * math.sin(np.deg2rad(deg_4))

    x_light5 = r_station * math.cos(np.deg2rad(deg_5))
    y_light5 = -r_station * math.sin(np.deg2rad(deg_5))

    x_light6 = r_station * math.cos(np.deg2rad(deg_6))
    y_light6 = -r_station * math.sin(np.deg2rad(deg_6))

    x_light7 = r_station * math.cos(np.deg2rad(deg_7))
    y_light7 = -r_station * math.sin(np.deg2rad(deg_7))

    x_light8 = r_station * math.cos(np.deg2rad(deg_8))
    y_light8 = -r_station * math.sin(np.deg2rad(deg_8))

    wordpoints_array = np.array([[x_light1, x_light2, x_light3, x_light4,x_light5, x_light6, x_light7, x_light8], \
                                 [y_light1, y_light2, y_light3, y_light4,y_light5, y_light6, y_light7, y_light8], \
                                 [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)#8 lights
    Intrin_mat = np.array([[kx,0,u0],[0,ky,v0],[0,0,1]],dtype=np.float32)
    Pw = wordpoints_array
    Pw_hom = np.concatenate((Pw,np.ones((1,Pw.shape[1]))),axis=0)
    #from world frame to camera frame
    R = R_euler
    RT = np.concatenate((R,T),axis = 1)
    RT_hom = np.concatenate((RT,np.array([[0,0,0,1]])),axis=0) #out 4*4
    Pc_hom = np.matmul(RT_hom,Pw_hom)# points in camera frame(homogeneous) 4*n points
    Pc = Pc_hom[0:3,:]
    Puv_hom = np.matmul(Intrin_mat,Pc)/Pc[2,:]
    Puv = Puv_hom[0:2,:]

    Img_gen = np.zeros([imgheight,imgwid],dtype=np.uint8)
    Puv_list = list(Puv.transpose().astype(np.int))
    for i,cor  in enumerate(Puv_list):
        cv2.circle(Img_gen,(cor[0],cor[1]),6,(255,255,255),-1)#cor[0]: column,cor[1]:row

    Puv_array=np.array(Puv_list)
    col_invalid =np.where(Puv_array[:,0]>imgwid)
    row_invalid = np.where(Puv_array[:, 1] > imgheight)
    if np.array(col_invalid).tolist() != [[]] or np.array(row_invalid).tolist() != [[]] or np.array(np.where(Puv_array[:, 1] < 0)).tolist() != [[]] or np.array(np.where(Puv_array[:, 0] < 0)).tolist() != [[]]:
        return Img_gen,()


    #bbox:
    polygon = Polygon(Puv_list)
    polygon_scale_ratio=1.5
    polygon_scale = shapely.affinity.scale(polygon, xfact=polygon_scale_ratio, yfact=polygon_scale_ratio,
                                           origin=polygon.centroid)  # rescaleimage_box
    bbox_scaled =shapely.geometry.box(polygon_scale.bounds[0],polygon_scale.bounds[1],polygon_scale.bounds[2],polygon_scale.bounds[3])
    image_box=shapely.geometry.box(0,0,imgwid,imgheight)#Makes a rectangular polygon from the provided bounding box values, with counter-clockwise order by default.
    box_res = bbox_scaled.intersection(image_box).bounds #(mincol,minrow,maxcol,maxrow)

    return Img_gen,box_res

def my_hard_stretch(featuremap):
    vmin = tf.reduce_min(featuremap)
    vmax = tf.reduce_max(featuremap)
    value = (featuremap-vmin)/(vmax-vmin+1e-8)
    return value

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    #input_paths = glob.glob(os.path.join(a.input_dir, "*_merge.jpg"))
    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted icnputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            #a_images = preprocess(raw_input[:,:width//2,:])
            #b_images = preprocess(raw_input[:,width//2:,:])
            orin_images = preprocess(raw_input[:,:width//3,:])
            bak_images = preprocess(raw_input[:,width//3:width*2//3,:])
            lightmask_images = preprocess(raw_input[:,width*2//3:,:])
            b_images = tf.concat([bak_images,lightmask_images],axis=2)
            b_images = b_images[:,:,:4]


    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        #inputs, targets = [b_images, a_images]
        inputs,targets = [b_images,orin_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    back_layers=[] #my backgroun stream layers
    lightmark_inputs = tf.expand_dims(generator_inputs[:, :, :, 3],axis=3)
    #my background stream=====================================================================
    back_inputs =generator_inputs[:,:,:,0:3]# need to check
    layer_specs = [
        #a.ngf//2,  # downstream: [batch, 256, 256, ngf/2] => [batch, 128, 128, ngf/2 ]
        a.ngf * 1,  # downstream: [batch, 128, 128, ngf/2] => [batch, 64, 64, ngf ]
        a.ngf * 2,  # downstream: [batch, 64, 64, ngf ] => [batch, 32, 32, ngf *2]
        a.ngf * 4,  # downstream: [batch, 32, 32, ngf *2] => [batch, 16, 16, ngf * 4]
        a.ngf * 4,  # downstream: [batch, 16, 16, ngf * 4] => [batch, 8, 8, ngf * 4]
        a.ngf * 4,  # downstream: [batch, 8, 8, ngf * 4] => [batch, 4, 4, ngf * 4]
        a.ngf * 4,  # downstream: [batch, 4, 4, ngf * 4] => [batch, 2, 2, ngf * 4]
    ]
    with tf.variable_scope("backstream_1"):#background stream
        output = gen_conv(back_inputs, a.ngf//2)
        output = batchnorm(output)
        back_layers.append(output)

    for out_channels in layer_specs:
        with tf.variable_scope("backstream_%d" % (len(back_layers) + 1)):
            convolved = gen_conv(back_layers[-1], out_channels)
            output = batchnorm(convolved)
            back_layers.append(output)

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(lightmark_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        back_layerid = num_encoder_layers-decoder_layer-2# number of corresponding background layer
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            decoder_firsthalf = output[:,:,:,:out_channels//2] #my operation
            decoder_lasthalf = output[:, :, :, out_channels // 2:]  # my operation
            back_stream = back_layers[back_layerid]
            output = tf.sigmoid(decoder_firsthalf)*back_stream+decoder_lasthalf*(1-tf.sigmoid(decoder_firsthalf))
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output_all = gen_deconv(rectified, generator_outputs_channels + 1)
        mask = output_all[:,:,:,-1]
        image_out = output_all[:,:,:,:-1]
        mask_stack = tf.stack([mask,mask,mask],axis=3)
        output = my_hard_stretch(mask_stack) * back_inputs + image_out * (1 - my_hard_stretch(mask_stack))

        output = tf.tanh(output)
        layers.append(output_all)
        layers.append(output)
    return layers


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        gen_layers_outputs = create_generator(inputs, out_channels)
        gen_layers13 = tf.squeeze(gen_layers_outputs[15])
        layer_unstack = tf.unstack(gen_layers13,axis=2)
        outputs = gen_layers_outputs[-1]

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        gen_layers_outputs=layer_unstack
    )





def main():

    myinput_np, mytarget_np,bbox = my_readimg_np('test2.jpg')
    if bbox==():#invalid setting
        return [],[],[],bbox
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    examples = load_examples()
    print("examples count = %d" % examples.count)

    g_blend = tf.Graph()
    with g_blend.as_default():
        myinputs_p  = tf.placeholder(tf.float32,shape=[1,256,256,4])# batch,height,width,channel,here is [1,256,256,4]
        mytargets_p = tf.placeholder(tf.float32,shape=[1,256,256,3])
        model = create_model(myinputs_p,mytargets_p)
        inputs = deprocess(myinputs_p)
        targets = deprocess(mytargets_p)
        outputs = deprocess(model.outputs)

        def convert(image):
            if a.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
                size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
                image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
                "feature_maps":model.gen_layers_outputs
            }

            my_display_fetches = {
                "inputs": converted_inputs,
                "targets": converted_targets,
                "outputs": converted_outputs,
            }

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            results = sess.run(my_display_fetches,feed_dict={myinputs_p:myinput_np,mytargets_p:mytarget_np})
            outputs_np = cv2.cvtColor(np.squeeze(results["outputs"]),cv2.COLOR_RGB2BGR)
            output_str = 'lx' + str(round(a.lx, 3)) + 'ly_' + str(round(a.ly, 3)) + 'lz_' + str(round(a.lz, 3)) + \
                         'br_' + str(round(a.betar, 3)) + 'bg_' + str(round(a.betag, 3)) + 'bb_' + str(
                round(a.betab, 3)) + \
                         'l_' + str(round(a.light, 3)) + 'g_' + str(round(a.g, 3)) + \
                         'h_' + str(round(a.heading, 3)) + 'p_' + str(round(a.pitch, 3)) + 'r_' + str(
                round(a.roll, 3)) + \
                         'tx_' + str(round(a.Tx, 3)) + 'ty_' + str(round(a.Ty, 3)) + 'tz_' + str(round(a.Tz, 3))
            cv2.imwrite(os.path.join('generated_img/',output_str+'.jpg'), outputs_np)
            print("rate", (time.time() - start) / max_steps)
            return np.squeeze(results["outputs"]),outputs_np,output_str,bbox#RGB,BGR,filename_base,bbox


if __name__=='__main__':
    time_start = time.time()
    output_str = 'lx' + str(round(a.lx, 3)) + 'ly_' + str(
        round(a.ly, 3)) + 'lz_' + str(round(a.lz, 3)) + \
                 'br_' + str(round(a.betar, 3)) + 'bg_' + str(
        round(a.betag, 3)) + 'bb_' + str(round(a.betab, 3)) + \
                 'l_' + str(round(a.light, 3)) + 'g_' + str(
        round(a.g, 3)) + \
                 'h_' + str(round(a.heading, 3)) + 'p_' + str(
        round(a.pitch, 3)) + 'r_' + str(round(a.roll, 3)) + \
                 'tx_' + str(round(a.Tx, 3)) + 'ty_' + str(
        round(a.Ty, 3)) + 'tz_' + str(round(a.Tz, 3))
    print(output_str + '   is running\n')

    # check first if out of range ====================================================
    kx = 633.5915  # written in many places. caution inconsistence
    ky = 695.2624
    u0 = 364.5163
    v0 = 234.3574
    imgwid = int(720)
    imgheight = int(576)
    pix_x = a.Tx * kx / a.Tz + u0  # from 3D world points to pixel
    pix_y = a.Ty * ky / a.Tz + v0
    wid_outrange = (pix_x > imgwid - 10 or pix_x < 0 + 10)
    height_outrange = (pix_y > imgheight - 10 or pix_y < 0 + 10)
    if wid_outrange or height_outrange:
        print(output_str, 'failed in range checking\n')
    # ===============================================================================
    RGB, BGR, filename_base, bbox = main()
    if bbox == ():  # invalid settings
        print(output_str + 'failed\n')
    time_end = time.time()
    print('one generation cost', time_end - time_start, '\n')







