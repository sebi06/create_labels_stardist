# -*- coding: utf-8 -*-

#################################################################
# File        : segmentation_stardist.py
# Author      : sebi06
#
# Kudos also to: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/docs/demo.ipynb
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import numpy as np
import logging

try:
    print("Trying to find tensorflow library ...")
    # silence tensorflow output
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    import tensorflow as tf

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    print("TensorFlow Version : ", tf.version.GIT_VERSION, tf.__version__)
except (ImportError, ModuleNotFoundError) as error:
    # Output expected ImportErrors.
    print(error.__class__.__name__ + ": " + error.msg)
    print("TensorFlow will not be used.")
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
from csbdeep.data import Normalizer, normalize_mi_ma


def segment_nuclei_stardist(
    img2d: np.ndarray,
    sdmodel: StarDist2D,
    axes: str = "YX",
    prob_thresh: float = 0.5,
    overlap_thresh: float = 0.3,
    overlap_label: Union[int, None] = None,
    blocksize: int = 1024,
    min_overlap: int = 128,
    n_tiles: Union[int, None] = None,
    norm_pmin: float = 1.0,
    norm_pmax: float = 99.8,
    norm_clip: bool = False,
    local_normalize: bool = True,
):
    """
    Segment nuclei in a 2D image using the StarDist2D model.

    Parameters:
        img2d (np.ndarray): 2D image array to be segmented.
        sdmodel (StarDist2D): Pre-trained StarDist2D model for segmentation.
        axes (str): Axes of the input image, default is "YX".
        prob_thresh (float): Probability threshold for object prediction, default is 0.5.
        overlap_thresh (float): Overlap threshold for non-maximum suppression, default is 0.3.
        overlap_label (Union[int, None]): Label for overlapping regions, default is None.
        blocksize (int): Size of the blocks for processing large images, default is 1024.
        min_overlap (int): Minimum overlap size for block processing, default is 128.
        n_tiles (Union[int, None]): Number of tiles for tiling large images, default is None.
        norm_pmin (float): Minimum percentile for normalization, default is 1.0.
        norm_pmax (float): Maximum percentile for normalization, default is 99.8.
        norm_clip (bool): Whether to clip values during normalization, default is False.
        local_normalize (bool): Whether to apply local normalization, default is True.
    Returns:
        np.ndarray: Segmented mask of the input image.
    """
    if local_normalize:
        # normalize whole 2d image
        img2d = normalize(
            img2d,
            pmin=norm_pmin,
            pmax=norm_pmax,
            axis=None,
            clip=norm_clip,
            eps=1e-20,
            dtype=np.float32,
        )

        normalizer = None

    if not local_normalize:
        mi, ma = np.percentile(img2d, [norm_pmin, norm_pmax])
        # mi, ma = image2d.min(), image2d.max()

        normalizer = MyNormalizer(mi, ma)

    # estimate blocksize
    max_dim_size = max(img2d.shape)
    blocksize = int(2 ** (np.round(np.log(max_dim_size) / np.log(2), 0) - 1))

    # define tiles
    if n_tiles is not None:
        if axes == "YX":
            tiles = (n_tiles, n_tiles)
        if axes == "YXC":
            tiles = (n_tiles, n_tiles, 1)
    if n_tiles is None:
        tiles = None

    # predict the instances of th single nuclei
    if max_dim_size >= 4096:
        mask2d, details = sdmodel.predict_instances_big(
            img2d,
            axes=axes,
            normalizer=normalizer,
            prob_thresh=prob_thresh,
            nms_thresh=overlap_thresh,
            block_size=blocksize,
            min_overlap=min_overlap,
            context=None,
            n_tiles=tiles,
            show_tile_progress=False,
            overlap_label=overlap_label,
            verbose=False,
        )
    if max_dim_size < 4096:
        mask2d, details = sdmodel.predict_instances(
            img2d,
            axes=axes,
            normalizer=normalizer,
            prob_thresh=prob_thresh,
            nms_thresh=overlap_thresh,
            n_tiles=tiles,
            # n_tiles=None,
            show_tile_progress=False,
            overlap_label=overlap_label,
            verbose=False,
        )

    return mask2d


def load_stardistmodel(modeltype: str = "Versatile (fluorescent nuclei)") -> StarDist2D:
    """Load an StarDist model from the web.

    Args:
        modeltype (str, optional): Name of the StarDist model to be loaded. Defaults to 'Versatile (fluorescent nuclei)'.

    Returns:
        StarDist2D: StarDist2D Model
    """

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # define and load the stardist model
    sdmodel = StarDist2D.from_pretrained(modeltype)

    return sdmodel


def stardistmodel_from_folder(
    modelfolder: str, mdname: str = "2D_dsb2018"
) -> StarDist2D:
    """Load an StarDist model from a folder.

    Args:
        modelfolder (str): Basefolder for the model folders.
        mdname (str, optional): Name of the StarDist model to be loaded. Defaults to '2D_dsb2018'.

    Returns:
        StarDist2D: StarDist2D Model
    """

    # workaround explained here to avoid errors
    # https://github.com/openai/spinningup/issues/16
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    sdmodel = StarDist2D(None, name=mdname, basedir=modelfolder)

    return sdmodel


class MyNormalizer(Normalizer):
    """
    A custom normalizer class that extends the Normalizer base class.
    Attributes:
        mi (float): The minimum value for normalization.
        ma (float): The maximum value for normalization.
    Methods:
        __init__(mi, ma):
            Initializes the MyNormalizer with the given minimum and maximum values.
        before(x, axes):
            Normalizes the input array `x` using the specified minimum (`mi`) and maximum (`ma`) values.
        after(*args, **kwargs):
            This method is not implemented and will always raise an assertion error.
        do_after:
            A property that always returns False, indicating that no post-processing is required.
    """

    def __init__(self, mi, ma):
        self.mi, self.ma = mi, ma

    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

    def after(*args, **kwargs):
        assert False

    @property
    def do_after(self):
        return False
