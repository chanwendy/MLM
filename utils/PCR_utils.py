# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from glob import glob
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)
import numpy as np
from torch.nn import MSELoss
from scipy import ndimage
from skimage.transform import resize
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import os
import random
from torch.utils.data import Dataset
from copy import copy
import logging
from skimage import measure
from scipy.special import softmax
import re
import imageio
import json
from batchgenerators.transforms.spatial_transforms import SpatialTransform, augment_spatial
import ipdb

tqdm, _ = optional_import("tqdm", name="tqdm")

__all__ = ["sliding_window_inference"]


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode), value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True  # whether the predictor's output is a tensor (instead of dict/tuple)

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat(
            [convert_data_type(inputs[win_slice], torch.Tensor)[0] for win_slice in unravel_slice]
        ).to(sw_device)
        seg_prob_out, x_cls = predictor(window_data, *args, **kwargs)  # batched patch segmentation

        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    final_output = final_output[0] if is_tensor_output else final_output  # type: ignore
    if isinstance(inputs, MetaTensor):
        final_output = convert_to_dst_type(final_output, inputs)[0]  # type: ignore
    return final_output, x_cls


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)



# npy loader part

def Tag(temp, index, tempC, num_component):
    tempC[index[0], index[1], index[2]] = num_component
    expand = []
    expand.append(index)
    count = 1
    while len(expand) > 0:
        temp_index = expand.pop()
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    new_index = [min(max(temp_index[0] + i, 0), temp.shape[0] - 1),
                                 min(max(temp_index[1] + j, 0), temp.shape[1] - 1),
                                 min(max(temp_index[2] + k, 0), temp.shape[2] - 1)]
                    if temp[new_index[0], new_index[1], new_index[2]] == 1 and tempC[
                        new_index[0], new_index[1], new_index[2]] == 0:
                        expand.append(new_index)
                        tempC[new_index[0], new_index[1], new_index[2]] = num_component
                        count += 1
                        # print(count)

    return


def check_connection(tumor_index, image):
    L = tumor_index.shape[0]
    temp = np.zeros_like(image)
    tempC = np.zeros_like(image)
    for i in range(L):
        temp[tumor_index[i, 0], tumor_index[i, 1], tumor_index[i, 2]] = 1

    num_component = 0
    for i in range(L):

        if tempC[tumor_index[i, 0], tumor_index[i, 1], tumor_index[i, 2]] == 0:
            num_component += 1
            Tag(temp, tumor_index[i, :], tempC, num_component)
    cc = []
    for i in range(L):
        cc.append(tempC[tumor_index[i, 0], tumor_index[i, 1], tumor_index[i, 2]])
    cc = np.array(cc)

    return cc


class BaseDataset(Dataset):
    """
    Base dataset class. Expects a list of dictionaries and a set of transforms
    to load data and transform it
    """

    def __init__(self, listdict, transforms=None):
        assert (type(listdict) == list), "Must provide a list of dicts to listdict"

        self._listdict = listdict
        self._transforms = transforms
        logging.debug('Dataset initialized with transform {}'.format(self._transforms))

    def __len__(self):
        return len(self._listdict)

    def __getitem__(self, idx):
        # here we assume the list dict is paths or image labels, we copy so as not
        # to modify the original list
        sample = copy(self._listdict[idx])
        if self._transforms:
            sample = self._transforms(sample)

        return sample


class BaseTransform(object):
    def __init__(self, fields):
        assert (isinstance(fields, (str, list))), "Fields must be a string or a list of strings"

        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields

    def __call__(self, sample):
        assert (isinstance(sample, dict)), "Each sample must be a dict"


class CopyField(BaseTransform):
    """
    Copy one field to another
    """

    def __init__(self, fields, to_field):
        super().__init__(fields)
        if len(fields) != 1:
            raise ValueError("Only provide one field for source")

        if isinstance(to_field, list):
            if len(to_field) != 1:
                raise ValueError("Only provide one field for destination")
        else:
            to_field = [to_field]

        self.to_field = to_field

    def __call__(self, data_dict):
        # to_field ['venous_origin']
        # print("to_field {}".format(self.to_field))
        # 复制input image原key("venous") 到 另一个key中（"venous_origin"）
        data_dict[self.to_field[0]] = copy(data_dict[self.fields[0]])
        # self.to_field[0] venous_origin
        # self.fields[0] venous
        # print("self.to_field[0] {}".format(self.to_field[0]))
        # print("self.fields[0] {}".format(self.fields[0]))

        return data_dict


class NiiLoader(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32, pre_set=None):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.pre_set = pre_set

    def __call__(self, data_dict):
        if not self.pre_set:
            self.pre_set = random.sample(self.fields, 2)

        data_dict['source'] = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0]]))).astype(self.dtype)
        data_dict['target'] = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1]]))).astype(self.dtype)
        data_dict['source_lung'] = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0] + '_lung']))).astype(self.dtype)
        data_dict['target_lung'] = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1] + '_lung']))).astype(self.dtype)
        if data_dict.get(self.pre_set[0] + '_stomach', None) and data_dict.get(self.pre_set[1] + '_stomach', None):
            data_dict['source_stomach'] = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0] + '_stomach']))).astype(
                self.dtype)
            data_dict['target_stomach'] = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1] + '_stomach']))).astype(
                self.dtype)
        return data_dict


class ReadNPY(BaseTransform):
    def __init__(self, root_dir='/', root_dir2='/', root_dir3=None):
        super().__init__(fields='sss')
        self.root_dir = root_dir
        self.root_dir2 = root_dir2
        self.root_dir3 = root_dir3

    def __call__(self, name):
        out_dict = {}
        A = np.load(os.path.join(self.root_dir, name))
        if not os.path.exists(os.path.join(self.root_dir2, name[0:-4] + '.npy')):
            B = np.zeros_like(A['labelV'])
            B[...] = A['labelV'][...]
            B[B > 0] = 13
        else:
            B = np.load(os.path.join(self.root_dir2, name[0:-4] + '.npy'))
        if self.root_dir3 is not None:
            qihang_A = np.load(os.path.join(self.root_dir3, name))

        out_dict['labelV'] = A['labelV'].astype(np.float32)
        out_dict['multi_organV'] = B.astype(np.float32)
        out_dict['name'] = name
        if self.root_dir3 is not None:
            out_dict['softV'] = qihang_A['pred'].astype(np.float32)
        else:
            out_dict['predV'] = A['predV'].astype(np.float32)
            out_dict['softV'] = np.argmax(out_dict['predV'], 0).astype(np.float32)
        print('read done')
        return out_dict


class NumpyLoader(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32, pre_set=None, load_mask=False):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.pre_set = pre_set
        self.load_mask = load_mask

    def __call__(self, data_dict):
        if self.pre_set is None:
            self.pre_set = random.sample(self.fields, min(2, len(self.fields)))
        if data_dict.get(self.pre_set[0], None):
            data_dict['source'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[0]])).astype(self.dtype)
        if self.load_mask:
            if data_dict.get(self.pre_set[0] + '_lung', None):
                data_dict['source_lung'] = np.load(
                    os.path.join(self.root_dir, data_dict[self.pre_set[0] + '_lung'])).astype(self.dtype)
            if data_dict.get(self.pre_set[0] + '_stomach', None):
                data_dict['source_stomach'] = (
                    np.load(os.path.join(self.root_dir, data_dict[self.pre_set[0] + '_stomach']))).astype(self.dtype)
        if len(self.pre_set) > 1:
            if data_dict.get(self.pre_set[1], None):
                data_dict['target'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[1]])).astype(
                    self.dtype)
            if self.load_mask:
                if data_dict.get(self.pre_set[1] + '_lung', None):
                    data_dict['target_lung'] = np.load(
                        os.path.join(self.root_dir, data_dict[self.pre_set[1] + '_lung'])).astype(self.dtype)
                if data_dict.get(self.pre_set[1] + '_stomach', None):
                    data_dict['target_stomach'] = (
                        np.load(os.path.join(self.root_dir, data_dict[self.pre_set[1] + '_stomach']))).astype(
                        self.dtype)

        return data_dict


class CropResize(BaseTransform):
    def __init__(self, fields, output_size, pad=32, shift=0):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.output_size = output_size
        self.pad = pad
        self.shift = shift

    def __call__(self, data_dict):
        # pad_width=32
        shift = self.shift
        for f in self.fields:
            if data_dict.get(f) is not None:
                if isinstance(data_dict.get(f + '_stomach_pred'), np.ndarray):
                    pred = data_dict.get(f + '_stomach_pred')
                    index = np.array(np.where(pred > 0)).T
                    bbox_max = np.max(index, 0)
                    bbox_min = np.min(index, 0)
                    center = (bbox_max + bbox_min) // 2
                    L = np.max(bbox_max - bbox_min)
                    pad_width = int(L * 0.1)
                    pred = pred[
                           max(center[0] - L // 2 - pad_width, 0):min(center[0] + L // 2 + pad_width, pred.shape[0]), \
                           max(center[1] - L // 2 - pad_width, 0):min(center[1] + L // 2 + pad_width, pred.shape[1]), \
                           max(center[2] - L // 2 - pad_width, 0):min(center[2] + L // 2 + pad_width, pred.shape[2])]
                    diff = list(L + pad_width * 2 - np.array(pred.shape))
                    axis_pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
                    pred = np.pad(pred, axis_pad_width)

                    data_dict[f + '_stomach_pred'] = resize(pred, self.output_size, order=0, anti_aliasing=False)
                else:
                    # 获取到stomach的坐标
                    # origin index shape is (3, 91166)
                    # print("origin index shape is {}".format(np.array(np.where(data_dict[f+'_stomach']>0)).shape))
                    index = np.array(np.where(data_dict[f + '_stomach'] > 0)).T
                    # after T index shape is (91166, 3)
                    # print("after T index shape is {}".format(np.array(np.where(data_dict[f + '_stomach'] > 0)).T.shape))
                    # print("stomach: ", data_dict[f+'_stomach'].shape)
                    # print("picture: ", data_dict[f].shape)
                    # print("index: ", index.shape)
                    if index.shape[0] > 0:
                        # 取第0维的最大值和最小值,其实为label中最大的坐标点和最小的坐标点
                        bbox_max = np.max(index, 0)
                        bbox_min = np.min(index, 0)
                        # 取中点
                        center = (bbox_max + bbox_min) // 2
                        # 取相差最大的边长
                        L = np.max(bbox_max - bbox_min)
                        pad_width = int(L * 0.1)
                    else:
                        center = np.array([64, 64, 64])
                        L = 32
                        pad_width = int(L * 0.1)
                img = data_dict.get(f)
                label = data_dict.get(f + '_stomach')
                data_dict['ori_shape'] = list(label.shape)
                # ori label shape is (213, 213, 186)
                # print("ori label shape is {}".format(label.shape))
                label = label[max(center[0] - L // 2 - pad_width + shift, 0):min(center[0] + L // 2 + pad_width + shift,
                                                                                 label.shape[0]), \
                        max(center[1] - L // 2 - pad_width + shift, 0):min(center[1] + L // 2 + pad_width + shift,
                                                                           label.shape[1]), \
                        max(center[2] - L // 2 - pad_width + shift, 0):min(center[2] + L // 2 + pad_width + shift,
                                                                           label.shape[2])]
                # ？
                diff = list(L + pad_width * 2 - np.array(label.shape))
                # [(x,y),(a,b),(c,d)]
                axis_pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
                # 填充
                label = np.pad(label, axis_pad_width)
                # pad label shape is (177, 177, 177)
                # print("pad label shape is {}".format(label.shape))
                data_dict['ori_shape'] += list(label.shape)
                # dict [213, 213, 186, 177, 177, 177] 前三个原维度，后三个pad后的维度
                # print(" dict {}".format(data_dict['ori_shape']))
                data_dict['ori_shape'] = np.array(data_dict['ori_shape'])
                img = img[max(center[0] - L // 2 - pad_width + shift, 0):min(center[0] + L // 2 + pad_width + shift,
                                                                             img.shape[0]), \
                      max(center[1] - L // 2 - pad_width + shift, 0):min(center[1] + L // 2 + pad_width + shift,
                                                                         img.shape[1]), \
                      max(center[2] - L // 2 - pad_width + shift, 0):min(center[2] + L // 2 + pad_width + shift,
                                                                         img.shape[2])]
                # img shape is (176, 176, 168)
                # print("img shape is {}".format(img.shape))
                diff = list(L + pad_width * 2 - np.array(img.shape))
                axis_pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
                img = np.pad(img, axis_pad_width)
                # print("1: ", data_dict[f].max(), data_dict[f].min())
                # print("2: ", data_dict[f].shape, self.output_size)
                data_dict[f] = resize(img, self.output_size)
                # print(self.output_size)
                # print("3: ", data_dict[f].max(), data_dict[f].min())
                data_dict[f + '_stomach'] = resize(label, self.output_size, order=0, anti_aliasing=False)
                # 最后输出的维度为（128，128，128）分别表示（H,W,S）

        return data_dict


class NumpyLoader_Multi(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32, load_mask=False, load_pred=False):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.load_mask = load_mask
        self.load_pred = load_pred

    def __call__(self, data_dict):
        for f in self.fields:
            if data_dict.get(f) is not None:
                data_dict[f] = np.load(os.path.join(self.root_dir, data_dict[f])).astype(self.dtype)
            if self.load_mask:
                if data_dict.get(f + '_stomach', None):
                    data_dict[f + '_stomach'] = np.load(
                        os.path.join(self.root_dir, data_dict[f + '_stomach'])).astype(self.dtype)
            if self.load_pred:
                if data_dict.get(f + '_stomach_pred', None):
                    data_dict[f + '_stomach_pred'] = np.load(
                        os.path.join(self.root_dir, data_dict[f + '_stomach_pred'])).astype(self.dtype)
        return data_dict


class NumpyLoader_Multi_merge(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', middle_path='/', dtype=np.float32, load_mask=False, load_pred=False,
                 load_pseudo=False, load_seg_npy=False, mask_index=None):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.middle_path = middle_path
        self.dtype = dtype
        self.load_mask = load_mask
        self.load_pred = load_pred
        self.load_pseudo = load_pseudo
        self.load_seg_npy = load_seg_npy
        self.mask_index = mask_index

    def __call__(self, input_string):
        data_dict = {}
        data_dict['id'] = ''.join(re.findall(r'\d+', input_string))
        # print("input_string is {}".format(input_string))    # input_string is stomach_0038/merge.npy
        # print("data_dict['id'] is  {}".format(data_dict['id']))     # data_dict['id'] is  0038
        # print("fields is {}".format(self.fields))   # fields is ['venous']
        # ipdb.set_trace()
        # score = json.load(open("/mnt/sdd/yaoyuan/VAE_segmentation/MaskData/score.json", 'r'))
        # aa = data_dict['id']
        # filename = os.path.join('compare/', f'{aa}_recon.pt')
        # torch.save(batch[label_key+'_recon_pred'][0], filename)
        normalfiles = glob("/HDD_data/MING/VAE/PCR_data/OD/ROInormal" + '/*.nii.gz')
        PCRniifiles = glob('/HDD_data/MING/VAE/PCR_data/OD/ROIPCR' + '/*.nii.gz')
        normalfiles.sort()
        PCRniifiles.sort()

        label = {}
        for file in normalfiles:
            label[re.findall(r'\d+', file)[0]] = 0
        for file in PCRniifiles:
            label[re.findall(r'\d+', file)[0]] = 1
        data_dict["cls_label"] = label[data_dict["id"]]
        # print(data_dict['id'])
        for f in self.fields:
            # print("f is {}".format(f))      # f is venous
            merge_data = np.load(os.path.join(self.root_dir, input_string))
            # if self.load_seg_npy:
            #     data_dict[f] = merge_data[0,1,...].astype(self.dtype)
            #     data_dict[f+'_stomach'] = merge_data[0,1,...].astype(self.dtype)
            #     data_dict[f+'_score'] = np.array([score[data_dict['id']]])
            #     continue
            # 0为 input_image  1为label
            data_dict[f] = merge_data[..., 0].astype(self.dtype)
            if self.load_mask:
                if self.mask_index is None:
                    data_dict[f + '_stomach'] = merge_data[..., 1].astype(self.dtype)
                else:

                    data_dict[f + '_stomach'] = np.zeros_like(merge_data[..., 1])
                    for label in self.mask_index:
                        # print("mask is {}".format(self.mask_index)) [[[0], 0], [[1], 1]]
                        # label =[[0], 0] or [[1], 1]
                        if not isinstance(label[0], list): label[0] = [label[0]]
                        for lab in label[0]:
                            # label[0] = [0] ->lab = 0 , label[1] = 0 上面zeros_like创建了一个与label一样大小的矩阵，接下来根据label的值填充矩阵，
                            # [merge_data[...,1]==lab] 一个一维数组，为lab的值具体来说就是将label中为0的位置设置为0， 为1的位置设置为1，[merge_data[...,1]==lab] 主要是确定索引，要设置哪里

                            data_dict[f + '_stomach'][merge_data[..., 1] == lab] = label[1]
                    data_dict[f + '_stomach'] = data_dict[f + '_stomach'].astype(self.dtype)
                    # print(data_dict[f+'_stomach'].shape)
            # data_dict = {
            # "id"： stomach_0038/merge.npy
            # "venous"： input_image
            # "venous_stomach": label
            # }
            if self.load_pseudo:
                filename = os.path.join(self.middle_path, '{}_pred.npy'.format(data_dict['id']))
                data_dict[f + '_stomach_pseudo'] = np.load(filename)
                # may introduce bug here
                # data_dict[f+'_stomach_pseudo'] = data_dict[f+'_stomach_pseudo'].cpu().detatch().numpy()[1]
            if self.load_pred:
                data_dict[f + '_stomach_pred'] = merge_data[..., 2].astype(self.dtype)
        return data_dict


class PadToSize(BaseTransform):
    """
    Pads numpy array to desired size, if necessary. IF array is larger or equal
    in size, do nothing. All padding is "right-sided" padding
    """

    def __init__(self, fields, size, pad_val=0, seg_pad_val=0, store_orig_size=True, random_subpadding=True,
                 load_mask=False):
        """
        size: the desired output size
        pad_val: the value to use for padding
        store_orig_size: if true, makes a new field storing the original size,
            which is sometimes needed for some applications

        """
        super().__init__(fields)
        self.size = np.array(size, dtype=np.int)
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.store_orig_size = store_orig_size
        self.random_subpadding = random_subpadding
        self.load_mask = load_mask

    def __call__(self, data_dict):
        start_idx = 0
        for field in self.fields:
            val = data_dict.get(field)
            val_lung = data_dict.get(field + '_lung')
            val_stomach = data_dict.get(field + '_stomach')
            if val is not None:
                orig_size = np.array(val.shape, dtype=np.int)
                # if any of the pad dims are greater than the orig_size, do pad
                if np.sum(np.greater(self.size, orig_size)) > 0:
                    diff = self.size - orig_size
                    diff[diff < 0] = 0
                    # set up right-side padding
                    pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
                    val = np.pad(val, pad_width, mode='constant',
                                 constant_values=self.pad_val)
                    data_dict[field] = val
                    if self.load_mask:
                        if val_lung is not None:
                            val_lung = np.pad(val_lung, pad_width, mode='constant',
                                              constant_values=self.seg_pad_val)
                            data_dict[field + '_lung'] = val_lung
                        if val_stomach is not None:
                            val_stomach = np.pad(val_stomach, pad_width, mode='constant',
                                                  constant_values=self.seg_pad_val)
                            data_dict[field + '_stomach'] = val_stomach

                if np.sum(np.greater(orig_size, self.size)) > 0:
                    # if not type(start_idx)==list: for consistency
                    # remove random supsampling when concate sup with unsup fxxk
                    maxes = list(orig_size - self.size)
                    if self.random_subpadding:
                        start_idx = [random.randint(0, max(maxy, 0)) for maxy in maxes]
                    else:
                        start_idx = [max(maxy, 0) for maxy in maxes]
                    '''
                    if self.load_mask:
                        if val_stomach is not None:
                            x,y,z = np.where(val_stomach>0)
                            start_idx[0] = min(max((np.max(x)+np.min(x))//2-self.size[0]//2,0),orig_size[0]-self.size[0])
                            #print('start ',start_idx[0])
                    '''

                    data_dict[field] = val[start_idx[0]:start_idx[0] + self.size[0],
                                       start_idx[1]:start_idx[1] + self.size[1],
                                       start_idx[2]:start_idx[2] + self.size[2]]
                    if self.load_mask:
                        if val_lung is not None:
                            val_lung = val_lung[start_idx[0]:start_idx[0] + self.size[0],
                                       start_idx[1]:start_idx[1] + self.size[1],
                                       start_idx[2]:start_idx[2] + self.size[2]]
                            data_dict[field + '_lung'] = val_lung
                        if val_stomach is not None:
                            val_stomach = val_stomach[start_idx[0]:start_idx[0] + self.size[0],
                                           start_idx[1]:start_idx[1] + self.size[1],
                                           start_idx[2]:start_idx[2] + self.size[2]]
                            data_dict[field + '_stomach'] = val_stomach
        return data_dict


class Reshape(BaseTransform):
    """
    Reshapes tensor without changing contents
    """

    def __init__(self, fields, reshape_view=None):
        super().__init__(fields)

        self._reshape_view = reshape_view

    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:

            if isinstance(data_dict.get(field), np.ndarray):
                if self._reshape_view is not None:
                    # field venous_stomach or venous_origin or  venous
                    # print("field {}".format(field))
                    # 3个field都是一样的shape data_dict shape is (1, 1, 128, 128, 128)
                    # print("data_dict shape is {}".format(data_dict[field].shape))
                    data_dict[field] = data_dict[field].reshape(self._reshape_view)
                    # 3个field after后的都是一样的after reshape data_dict shape is (1, 128, 128, 128)
                    # print("after reshape data_dict shape is {}".format(data_dict[field].shape))
                else:
                    # print("field {}".format(field))
                    # print("data_dict shape is {}".format(data_dict[field].shape))
                    # [-1,1]+list(data_dict[field].shape) ->[-1, 1, 128, 128, 128]
                    data_dict[field] = data_dict[field].reshape([-1, 1] + list(data_dict[field].shape))
                    # data_dict shape is (1, 1, 128, 128, 128) 扩充到5维，（B,  ,H ,W,S）
                    # print("data_dict shape is {}".format(data_dict[field].shape))
        return data_dict


class ExtendSqueeze(BaseTransform):
    """
    Reshapes tensor without changing contents
    """

    def __init__(self, fields, dimension=-1, mode=1):
        super().__init__(fields)

        self.dimension = dimension
        self.mode = mode

    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:

            if isinstance(data_dict.get(field), np.ndarray):
                if self.mode == 1:
                    data_dict[field] = np.expand_dims(data_dict[field], axis=self.dimension)
                if self.mode == 0:
                    data_dict[field] = np.squeeze(data_dict[field], axis=self.dimension)
        return data_dict


class Clip(BaseTransform):
    """
    Will clip numpy arrays and pytorch tensors
    """

    def __init__(self, fields, new_min=0.0, new_max=1.0):
        """
        new_min: min value to clip to
        new_max: max value to clip to
        """
        super().__init__(fields)

        self._new_min = new_min
        self._new_max = new_max

    def __call__(self, data_dict):

        for field in self.fields:
            if data_dict.get(field) is not None:
                val = data_dict[field]
                # check if numpy or torch.Tensor, and call appropriate method
                if isinstance(val, torch.Tensor):
                    data_dict[field] = torch.clamp(val, self._new_min, self._new_max)
                else:
                    data_dict[field] = np.clip(val, self._new_min, self._new_max)

        return data_dict


class Binarize(BaseTransform):
    """
    Binarize numpy arrays, or torch.tensors. Note, if doing it to
    torch.Tensor, this will copy to cpu and perform numpy operation
    """

    def __init__(self, fields, threshold=0.5, new_min=0, new_max=1, dtype=np.float32):
        """
        threshold: threshold value
        new_min: new value for values below threshold
        new_max: new value for values greater or equal to threshold
        dtype: dtype of resulting array
        """
        super().__init__(fields)
        self._threshold = threshold
        self._new_min = new_min
        self._new_max = new_max
        self.dtype = dtype

    def __call__(self, data_dict):
        for field in self.fields:
            if data_dict.get(field) is not None:
                val = data_dict[field]
                is_torch = False
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                    is_torch = True

                data_dict[field] = np.where(val >= self._threshold,
                                            self._new_max, self._new_min).astype(self.dtype)

                # convert back to tensor if needed
                if is_torch:
                    data_dict[field] = torch.from_numpy(data_dict[field])
        return data_dict


class CenterIntensities(BaseTransform):
    """
    Transform that subtracts by a subtrahend and divides by a divisor, most
    often done to whiten data by subtracting the mean and dividing by the std
    deviation.

    Note, this class assumes the pytorch shape conventions:
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    which consider the first dimension (index 0) to be the channel dimension.
    """

    def __init__(self, fields, subtrahend, divisor=1.0):
        """
        Args:
            fields: fields to apply centering
            subtrahend: the subtrahend used to subtract, if a list then subtraction
                is performed per channel
            divisor: sames as subtrahend, but specifies the divisor.
        """
        super().__init__(fields)

        # convert any lists to np.arrays, with an extra singleton dimension
        # to allow broadcasting
        if isinstance(divisor, list):
            divisor = np.array(divisor)
            divisor = np.expand_dims(divisor, 1)
        if isinstance(subtrahend, list):
            subtrahend = np.array(subtrahend)
            subtrahend = np.expand_dims(subtrahend, 1)
        self.subtrahend = subtrahend
        self.divisor = divisor

    def __call__(self, data_dict):

        for field in self.fields:
            if data_dict.get(field) is not None:
                old_shape = data_dict[field].shape
                # old shape is (1, 1, 128, 128, 128)
                # print("old shape is {}".format(old_shape))
                # reshape val, to allow broadcasting over 2D, 3D, or nd data
                val = data_dict[field].reshape((data_dict[field].shape[0], -1))
                # val shape is (1, 2097152)
                # print("val shape is {}".format(val.shape))
                # perform centering
                val -= self.subtrahend
                val /= self.divisor
                data_dict[field] = val.reshape(old_shape)
                # data_dict[field] shape (1, 1, 128, 128, 128)
                # print("data_dict[field] shape {}".format(data_dict[field].shape))

        return data_dict


class image_resize:
    def __init__(self, fields, target_size=64):
        self.fields = fields
        self.target_size = target_size

    def __call__(self, data_dict):

        for field in self.fields:
            if data_dict.get(field) is not None:
                if len(field.split('_')) > 1:
                    data_dict[field] = resize(data_dict[field], (self.target_size, 256, 256), order=0).astype(
                        np.float32)
                else:
                    data_dict[field] = resize(data_dict[field], (self.target_size, 256, 256), preserve_range=True,
                                              anti_aliasing=True).astype(np.float32)
        return data_dict


def load_NII(data_path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(data_path)
    vol = reader.Execute()

    return vol


def get_synthesis_mask(data_dict):
    field = 'venous'
    mask_bone_new = data_dict[field] > 200
    mask_bone_new = binary_dilation(mask_bone_new, iterations=2)

    mask_bowel_NC = np.zeros(data_dict[field].shape)
    mask_bowel_NC[data_dict[field] < 0] = 1
    data_dict[field + '_syn_mask'] = ((1 - mask_bowel_NC) * (1 - mask_bone_new)).astype(np.float32)
    return data_dict


def align_volume(data_dict, model, iterations=1):
    with torch.no_grad():
        model = model.cuda()
        for _ in range(iterations):
            data_dict = model(data_dict)
            if isinstance(data_dict[model.out_key], list):
                out_image = data_dict[model.out_key][0]
            else:
                out_image = data_dict[model.out_key]
            data_dict[model.source_key] = out_image
    orig_z = data_dict['arterial_original'].shape[0]
    print('orig_z', orig_z)
    data_dict['dfield'] = data_dict['dfield'][:, :, :orig_z, :]
    return data_dict


# def deform_volume(dfield, sitk_image):
#     volume = sitk.GetArrayViewFromImage(sitk_image)
#     volume = torch.from_numpy(volume.astype(np.float32)).cuda()
#     d, h, w = volume.shape
#     grid = create_grid(d, h, w)
#     grid = grid.cuda()
#     volume = volume.unsqueeze(0)
#     volume = volume.unsqueeze(0)

#     dfield = torch.nn.functional.interpolate(dfield, size=(d,h,w), mode='trilinear')
#     new_image = deform_image(volume, dfield, grid, (d,h,w))
#     new_image = sitk.GetImageFromArray(new_image.cpu().detach()[0,0,:])
#     new_image.CopyInformation(sitk_image)
#     return new_image


def create_grid_images(source, target, source_aligned, save_folder, slice_num=20, min_win=-280, max_win=420):
    """
    Given source, target, and aligned source volumes, save a slice of each, plus a checkerboard
    version overlaying the source over the target and the aligned source over the target. Saves
    them as .png files. Assumes slices are 512x512, if not then it will upsample them

    Args:
        source (np.array): source volume
        target (np.array): target volume
        source_aligned (np.array): algined source volume
        save_folder (str): location where you want png images saved
        slice_num (int): the slice number of the volume you want to compare
        min_win, max_win: the window to use for saving the image, as min and max intensities
    """

    s_np = sitk.GetArrayViewFromImage(source)
    t_np = sitk.GetArrayViewFromImage(target)
    sa_np = sitk.GetArrayViewFromImage(source_aligned)

    s_np = s_np[slice_num, :]
    t_np = t_np[slice_num, :]
    sa_np = sa_np[slice_num, :]

    if s_np.shape[0] != 512:
        s_np = resize(s_np, [512, 512])
    if t_np.shape[0] != 512:
        t_np = resize(t_np, [512, 512])
    if sa_np.shape[0] != 512:
        sa_np = resize(sa_np, [512, 512])

    # create checkerboard binary mask using the kronecker product
    checkerboard = np.kron([[1, 0] * 16, [0, 1] * 16] * 16, np.ones((16, 16)))

    # use the mask to create an overlayed image
    orig_check = s_np * checkerboard + (1 - checkerboard) * t_np
    align_check = sa_np * checkerboard + (1 - checkerboard) * t_np

    # save images
    all_images = [s_np, t_np, sa_np, orig_check, align_check]
    suffixes = ['source.png', 'target.png', 'source_align.png', 'orig_check.png', 'align_check.png']
    for image, suffix in zip(all_images, suffixes):
        image = np.clip(image, min_win, max_win)
        image = image - min_win

        image = image / (max_win - min_win)
        image = image * 255
        image = image.astype(np.uint8)
        cur_path = os.path.join(save_folder, suffix)
        imageio.imwrite(cur_path, image)


def predict_vol(net, v):
    # v = sitk.GetArrayFromImage(v_sitk)
    net.eval()

    # Step 1: CNN segmentation
    (_, _, zSize, ySize, xSize) = v.shape
    print(v.shape)
    # v = resize(v, [zSize, 512, 512],preserve_range=True, anti_aliasing=True)

    output_prob = np.zeros(shape=(zSize, 512, 512), dtype=np.float32)
    output_mask = np.zeros(shape=(zSize, 512, 512), dtype=np.int8)

    for k in range(0, zSize):
        input_data = v[:, :, k, :, :]
        # input_data = input_data.type(torch.cuda.FloatTensor)
        # input_data = input_data.cuda()

        with torch.no_grad():
            output_data = net(input_data)
            output_probs = output_data[4].squeeze(0).squeeze(0)  # predict lung segmentation

        output_prob[k, :, :] = output_probs.cpu()
        output_mask[k, :, :] = output_probs.cpu() > 0.5

    output_mask_sitk = sitk.GetImageFromArray(output_mask)  # save mask
    # output_mask_sitk.SetOrigin(v_sitk.GetOrigin())
    # output_mask_sitk.SetDirection(v_sitk.GetDirection())
    # output_mask_sitk.SetSpacing(v_sitk.GetSpacing())

    # Step 2: connected component post-processing
    componentFilter = sitk.ConnectedComponentImageFilter()
    componentFilter.SetFullyConnected(True)
    obj_label = componentFilter.Execute(output_mask_sitk)
    obj_count = componentFilter.GetObjectCount()

    relabelFilter = sitk.RelabelComponentImageFilter()
    relabelFilter.SortByObjectSizeOn()
    obj_relabel = relabelFilter.Execute(obj_label)
    label_count = relabelFilter.GetSizeOfObjectsInPixels()
    obj_recount = relabelFilter.GetNumberOfObjects()
    print(obj_recount, label_count)

    relabel_data = sitk.GetArrayFromImage(obj_relabel)

    for index in range(0, obj_recount):
        if label_count[index] < 10000 or index > 1:
            real_index = index + 1
            relabel_data[relabel_data == real_index] = 0

    relabel_data[relabel_data > 0] = 1

    relabel_mask = sitk.GetImageFromArray(relabel_data)
    # relabel_mask.SetOrigin(v_sitk.GetOrigin())
    # relabel_mask.SetSpacing(v_sitk.GetSpacing())
    # relabel_mask.SetDirection(v_sitk.GetDirection())

    return relabel_mask, v.cpu().numpy()


def mutual_information_3d(x, y, sigma=1, normalized=True):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)
    EPS = np.finfo(float).eps

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2)))

    return mi


def plot_slides(v, display_type='TB'):
    # Plot the slides
    d, h, w = v.shape
    side_w = int(np.ceil(np.sqrt(d)))
    side_h = int(np.ceil(float(d) / side_w))

    board = np.zeros(((h + 1) * side_h, (w + 1) * side_w, 3))

    if display_type == 'TB':
        MIN = 1 * np.min(v)  # Rescale the value of voxels into [0, 1], as unsigned byte
        MAX = 1 * np.max(v)
        v_n = (v - MIN) / (MAX - MIN)

        for i in range(side_h):
            for j in range(side_w):
                if i * side_w + j >= d:
                    break
                for k in range(3):
                    board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), k] = v_n[
                                                                                                       i * side_w + j,
                                                                                                       :, :]

    if display_type == 'CV2':
        MIN = 1 * np.min(v)  # Rescale the value of voxels into [0, 255], as unsigned byte
        MAX = 1 * np.max(v)
        v_n = (v - MIN) / (MAX - MIN)
        v_n = (v_n * 255).astype(int)

        for i in range(side_h):
            for j in range(side_w):
                if i * side_w + j >= d:
                    break
                for k in range(3):
                    board[(h + 1) * i + 1: (h + 1) * (i + 1), (w + 1) * j + 1: (w + 1) * (j + 1), k] = v_n[
                                                                                                       i * side_w + j,
                                                                                                       :, :]
        board = board.astype(int)

    # Return a 2D array representing the image pixels
    return board


def standard_loss(data_dict, do_mask=True, source_key='align_arterial', target_key='venous',
                  mask_key='venous_reg_mask'):
    """
    just computes the MSELoss between two images, but uses a mask to specify the image regions
    that should be used to compute the loss on
    """
    loss_layer = MSELoss()
    deformed_ims = data_dict[source_key]

    # here we compute loss on all outputs, which can happen for PHNN or PSNN models
    # if we have only one output, we make it a list.
    # e.g., if our model is PHNN or PSNN
    if not isinstance(deformed_ims, list):
        deformed_ims = [deformed_ims]
    standard_loss_sum = 0
    mask = data_dict[mask_key]
    for im in deformed_ims:
        if do_mask:
            dummy_arterial = mask * im + \
                             (1 - mask) * data_dict[target_key]
            data_dict['dummy_align_venous'] = dummy_arterial
        else:
            data_dict['dummy_align_venous'] = im
        standard_loss_sum += loss_layer(data_dict[target_key], data_dict['dummy_align_venous'])

    return standard_loss_sum


def smoothness_loss(data_dict):
    smooth_loss = data_dict['smooth_dform'].mean()
    return smooth_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total: {}".format(total_num))
    print("Trainable: {}".format(trainable_num))
    return {'Total': total_num, 'Trainable': trainable_num}


class MySpatialTransform(SpatialTransform):
    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1):

        super().__init__(patch_size, patch_center_dist_from_border, do_elastic_deform, alpha, sigma, do_rotation,
                         angle_x, angle_y, angle_z, do_scale, scale, border_mode_data, border_cval_data, order_data,
                         border_mode_seg, border_cval_seg, order_seg, random_crop, data_key, label_key, p_el_per_sample,
                         p_scale_per_sample, p_rot_per_sample, independent_scale_for_each_axis, p_rot_per_axis)

    def __call__(self, data_dict):
        data = data_dict.get(self.data_key)
        # data is (1, 1, 128, 128, 128)
        # print("data is {}".format(data.shape))
        # label
        seg = data_dict.get(self.label_key)
        # label is (1, 1, 128, 128, 128)
        # print("label is {}".format(seg.shape))

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                # 获取到patch的大小
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis)
        # print("ret_val is {}".format(ret_val))    # 返回一个tuple 第一个元素为img，第二个为label
        data_dict[self.data_key] = ret_val[0]
        # img after augment_spatial (1, 1, 128, 128, 128)
        # print("img after augment_spatial {}".format(ret_val[0].shape))
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
            # label after augment_spatial (1, 1, 128, 128, 128)
            # print("label after augment_spatial {}".format(ret_val[1].shape))

        return data_dict

