import functools
import json
import logging

import torch
import numpy as np
import PIL
from PIL import Image

import skimage
from skimage.exposure import match_histograms

from request_models import InpaintingRequest
from utils import size_from_aspect_ratio

logger = logging.getLogger(__name__)

# Outpainting noise
#
# https://github.com/parlance-zz/g-diffuser-bot/tree/g-diffuser-bot-diffuserslib-beta
# https://www.reddit.com/r/StableDiffusion/comments/xbjnnu/huge_outpainting_in_1_step_without_erased_colors/
#
DEFAULT_RESOLUTION = (512, 512)
RESOLUTION_GRANULARITY = 64
MAX_RESOLUTION = (512, 512)

def _valid_resolution(width, height, init_image=None): # cap max dimension at max res and ensure size is 
                                                       # a correct multiple of granularity while
                                                       # preserving aspect ratio (best we can anyway)
    global RESOLUTION_GRANULARITY
    global DEFAULT_RESOLUTION
    global MAX_RESOLUTION
    
    if not init_image:
        if not width: width = DEFAULT_RESOLUTION[0]
        if not height: height = DEFAULT_RESOLUTION[1]
    else:
        if not width: width = init_image.size[0]
        if not height: height = init_image.size[1]
        
    aspect_ratio = width / height 
    if width > MAX_RESOLUTION[0]:
        width = MAX_RESOLUTION[0]
        height = int(width / aspect_ratio + .5)
    if height > MAX_RESOLUTION[1]:
        height = MAX_RESOLUTION[1]
        width = int(height * aspect_ratio + .5)
        
    width = int(width / float(RESOLUTION_GRANULARITY) + 0.5) * RESOLUTION_GRANULARITY
    height = int(height / float(RESOLUTION_GRANULARITY) + 0.5) * RESOLUTION_GRANULARITY
    if width < RESOLUTION_GRANULARITY: width = RESOLUTION_GRANULARITY
    if height < RESOLUTION_GRANULARITY: height = RESOLUTION_GRANULARITY

    return width, height
    
# def _get_tmp_path(file_extension):
#     global TMP_ROOT_PATH
#     try: # try to make sure temp folder exists
#         pathlib.Path(TMP_ROOT_PATH).mkdir(exist_ok=True)
#     except Exception as e:
#         print("Error creating temp path: '" + TMP_ROOT_PATH + "' - " + str(e))
#     return TMP_ROOT_PATH + "/" + str(uuid.uuid4()) + file_extension

# def _save_debug_img(np_image, name):
#     global DEBUG_MODE
#     if not DEBUG_MODE: return
#     global TMP_ROOT_PATH
    
#     image_path = TMP_ROOT_PATH + "/_debug_" + name + ".png"
#     if type(np_image) == np.ndarray:
#         if np_image.ndim == 2:
#             mode = "L"
#         elif np_image.shape[2] == 4:
#             mode = "RGBA"
#         else:
#             mode = "RGB"
#         pil_image = PIL.Image.fromarray(np.clip(np_image*255., 0., 255.).astype(np.uint8), mode=mode)
#         pil_image.save(image_path)
#     else:
#         np_image.save(image_path)
    
def _dummy_checker(images, **kwargs): # replacement func to disable safety_checker in diffusers
    return images, False

def _factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]
    
def _get_grid_layout(num_samples):
    factors = _factorize(num_samples)
    median_factor = factors[len(factors)//2]
    columns = median_factor
    rows = num_samples // columns
    
    return (rows, columns)
    
def _image_grid(imgs, layout): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid

# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
    if data.ndim > 2: # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def _ifft2(data):
    if data.ndim > 2: # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def _get_gaussian_window(width, height, std=3.14, mode=0):

    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))
    
    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    for y in range(height):
        fy = (y / height * 2. - 1.) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x**2+fy**2) * std)
        else:
            window[:, y] = (1/((x**2+1.) * (fy**2+1.))) ** (std/3.14) # hey wait a minute that's not gaussian
            
    return window

def _get_masked_window_rgb(np_mask_grey, hardness=1.):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:,:,c] = hardened[:]
    return np_mask_rgb

"""
 Explanation:
 Getting good results in/out-painting with stable diffusion can be challenging.
 Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.
 Provided here is my take on a potential solution to this problem.
 
 By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation of each feature scale in the unmasked src.
 Shaping the init/seed noise for in/outpainting to the same distribution of feature scales, orientations, and positions increases output coherence
 by helping keep features aligned. This technique is applicable to any continuous generation task such as audio or video, each of which can
 be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For multi-channel data such as color
 or stereo sound the "color tone" or histogram of the seed noise can be matched to improve quality (using scikit-image currently)
 This method is quite robust and has the added benefit of being fast independently of the size of the out-painted area.
 The effects of this method include things like helping the generator integrate the pre-existing view distance and camera angle.
 
 np_src_image is a float64 np array of shape [width,height,3] ( range 0..1)
 np_mask_rgb is a float64 np array of shape [width,height,3] ( range 0..1)
 noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means higher detail (range > 0, default 1.)
 color_variation controls how much freedom is allowed for the colors/palette of the out-painted area (range 0..1, default 0.01)
 returns shaped noise for blending into the src image with the supplied mask ( [width,height,3] range 0..1 )
 
 The returned mask should be blended strongly into the src image, and mask hardening can improve the results.
 
 This code is provided as is under the Unlicense (https://unlicense.org/)
 Although you have no obligation to do so, if you found this code helpful please find it in your heart to credit me.
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 This code is part of a new branch of a discord bot I am working on integrating with diffusers (https://github.com/parlance-zz/g-diffuser-bot)
 
"""
def _get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation): 

    #global DEBUG_MODE
    #global TMP_ROOT_PATH
    
    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2)/3.) 
    np_src_grey = (np.sum(np_src_image, axis=2)/3.) 
    all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 0.5
    ref_mask = np_mask_grey < 0.5
    
    windowed_image = _np_src_image * (1.-_get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb# / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
    #windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    #_save_debug_img(windowed_image, "windowed_src_img")
    
    src_fft = _fft2(windowed_image) # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    #_save_debug_img(src_dist, "windowed_src_dist")
    
    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2)/3.) 
    noise_rgb *=  color_variation # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:,:,c] += (1. - color_variation) * noise_grey
        
    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:,:,c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:,:,:] = np.absolute(shaped_noise_fft[:,:,:])**2 * (src_dist ** noise_q) * src_phase # perform the actual shaping
    
    brightness_variation = 0.#color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.
    
    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    
    #shaped_noise -= np.min(shaped_noise)
    shaped_noise = np.clip(shaped_noise/ np.max(shaped_noise), 0., 1.)
    #shaped_noise_blended = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
    #shaped_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise_blended[all_mask,:], contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
    shaped_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], contrast_adjusted_np_src[ref_mask,:], channel_axis=1)
    shaped_noise = np.clip(shaped_noise / np.max(shaped_noise), 0., 1.)
    #_save_debug_img(shaped_noise, "shaped_noise")
    
    matched_noise = np.zeros((width, height, num_channels))
    matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    matched_noise[all_mask,:] = skimage.exposure.match_histograms(matched_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    #matched_noise[ref_mask,:] = skimage.exposure.match_histograms(matched_noise[ref_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    #_save_debug_img(matched_noise, "matched_noise")
    
    """
    todo:
    color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be param controlled
    """
    
    return np.clip(matched_noise, 0., 1.) 

class ParlanceZzNoise:
    def __init__(
        self
    ):
        super().__init__()

    def applyTo(self, request: InpaintingRequest, source_image: Image.Image, mask: Image.Image): 
        aspect_ratio = source_image.width / source_image.height
        size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)
        noise_q = request.noise_q
        color_variation = request.color_variation
        mask_blend_factor = request.mask_blend_factor
        seed = request.seed

        if seed:
            np.random.seed(seed)

        # TODO figure out expected resolution in noise procedure
        #      this completely breaks with ratios that aren't 1
        width = 512 #source_image.width
        height = 512 #source_image.height

        #DEFAULT_RESOLUTION = (width, height)
        #MAX_RESOLUTION = (width, height)    

        # Apply noise similar to the existing colors (outpainting)
        if request.parlance_zz_noise:
            logger.info("Parlance ZZ noise")
            np_init = (np.asarray(source_image.convert("RGB"))/255.0).astype(np.float64)
            np_mask_rgb = (np.asarray(mask.convert("RGB"))/255.0).astype(np.float64)

            noise_rgb = _get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
            final_mask = np.clip(np_mask_rgb + 0.0, 0., 1.)
            blend_mask_rgb = (final_mask ** mask_blend_factor)
            noised = np_init[:] * (1. - blend_mask_rgb) + noise_rgb * blend_mask_rgb
            
            # one last thing, gotta colorize the noise from src while preserving vector mag of blended noise img
            #"""
            
            np_init2 = np.clip(np_init[:] + np.random.normal(0, 1/16., (width, height, 3)), 0., 1.)
            np_mask_rgb2 = np.clip(np_mask_rgb[:] + np.random.normal(0, 1/16., (width, height, 3)), 0., 1.)
            np_init_mag_rgb = np.zeros((width, height, 3))
            np_init_mag_rgb[:,:,0] = np.sum(np_init2**2, axis=2) ** 0.5
            np_init_mag_rgb[:,:,1] = np_init_mag_rgb[:,:,0]
            np_init_mag_rgb[:,:,2] = np_init_mag_rgb[:,:,0]
            np_init_mag_rgb[np.where(np_init_mag_rgb <= 1e-8)] = 1.
            
            noised *= ((np_init2[:] ** 0.5) / np_init_mag_rgb ) ** (1. - np.clip(np_mask_rgb2*1.1, 0., 1.))
            noised = np_init[:] * (1. - np_mask_rgb2) + noised * np_mask_rgb2

            source_image = PIL.Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")    

        return source_image
