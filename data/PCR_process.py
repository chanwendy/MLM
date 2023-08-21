import json
import numpy as np
import os.path as path
import nibabel as nib
import os
import glob
import ipdb
import SimpleITK as sitk
import pydicom
from skimage.transform import resize
from tqdm import tqdm
import re

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return dirs,files  # 当前路径下所有非目录子文件,列表


PCRnormal_path = "/HDD_data/MING/VAE/PCR_data/OD/normalnii"
PCRnormal_label_path = "/HDD_data/MING/VAE/PCR_data/OD/normalonlyROI"
PCR_path = "/HDD_data/MING/VAE/PCR_data/OD/PCRnii"
PCR_label_path = "/HDD_data/MING/VAE/PCR_data/OD/PCRonlyROI"
to_path = "/HDD_data/MING/VAE/PCR_data/OD/npyfile_32p"

# normal data
normalnames = glob.glob(path.join(PCRnormal_path, '*.gz'))
normalnames.sort()
normal_label_names = glob.glob(path.join(PCRnormal_label_path, '*.gz'))
normal_label_names.sort()
# PCR data
PCRnames = glob.glob(path.join(PCR_path, '*.gz'))
PCRnames.sort()
PCR_label_names = glob.glob(path.join(PCR_label_path, '*.gz'))
PCR_label_names.sort()


pad = [32,32,32]
for i in tqdm(range(len(normalnames))):
    imgname = normalnames[i]
    labelname = normal_label_names[i]
    if re.findall(r'\d+', imgname) == re.findall(r'\d+', labelname):
        image = nib.load(imgname)
        spacing = image.affine[[0, 1, 2], [0, 1, 2]]

        # deciding the direction
        # 将spacing 值转为1或-1，
        ind = ((-spacing > 0) - 0.5) * 2
        image = image.get_data()
        # 将0 1 维度对换 意义是什么-》旋转90度
        image = np.transpose(image, [1, 0, 2])
        image = image[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]

        # resample to 1mm
        new_size = (np.array(image.shape) * np.abs(spacing)).astype(np.int)
        image = resize(image.astype(np.float64), new_size)

        label = nib.load(labelname)
        spacing = label.affine[[0, 1, 2], [0, 1, 2]]
        label = label.get_data()
        # ipdb.set_trace()
        label = np.transpose(label, [1, 0, 2])
        ind = ((-spacing > 0) - 0.5) * 2
        label = label[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]
        label = resize(label.astype(np.float64), new_size, anti_aliasing=False, order=0)
        print(imgname, 'loaded', new_size, spacing)

        # get the bounding box of foreground
        # 获取到器官位置 的index
        tempL = np.array(np.where(label > 0))
        print(tempL[0].shape)
        # bbox 每一个维度的最小值和最大值区间
        bbox = np.array([[max(0, np.min(tempL[0]) - pad[0]), min(label.shape[0], np.max(tempL[0]) + pad[0])], \
                         [max(0, np.min(tempL[1]) - pad[1]), min(label.shape[1], np.max(tempL[1]) + pad[1])], \
                         [max(0, np.min(tempL[2]) - pad[2]), min(label.shape[2], np.max(tempL[2]) + pad[2])]])
        # 每一个维度的中心点
        center = np.mean(bbox, 1).astype(int)
        # 最大值 - 最小值获得bbox长度
        bbL = bbox[:, 1] - bbox[:, 0]
        # 取最长的那条边
        L = int(np.max(bbox[:, 1] - bbox[:, 0]))
        print(L)

        # extract a cubic box(立方体) that contain all the foreground
        # 取出label 中PANCREAS的部分
        out = \
            image[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
            max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
            max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]
        Shape = list(out.shape)
        Shape.append(2)
        print(Shape)
        Out_img = out
        Out_label = \
            label[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
            max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
            max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]

        path_prefix = path.join(to_path, path.split(imgname)[1].split(".")[0])
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        np.save(path.join(path_prefix, 'img.npy'), Out_img.astype(np.int16))
        np.save(path.join(path_prefix, 'label.npy'), Out_label.astype(np.int8))
        # np.stack((x,y), axis=-1) 在最后新增一个维度，并在最后一个维度进行拼接， axis默认=0
        np.save(path.join(path_prefix, 'merge.npy'), np.stack((Out_img, Out_label), axis=-1).astype(np.int16))


for i in tqdm(range(len(PCRnames))):
    imgname = PCRnames[i]
    labelname = PCR_label_names[i]
    if re.findall(r'\d+', imgname) == re.findall(r'\d+', labelname):
        image = nib.load(imgname)
        spacing = image.affine[[0, 1, 2], [0, 1, 2]]

        # deciding the direction
        # 将spacing 值转为1或-1，
        ind = ((-spacing > 0) - 0.5) * 2
        image = image.get_data()
        # 将0 1 维度对换 意义是什么-》旋转90度
        image = np.transpose(image, [1, 0, 2])
        image = image[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]

        # resample to 1mm
        new_size = (np.array(image.shape) * np.abs(spacing)).astype(np.int)
        image = resize(image.astype(np.float64), new_size)

        label = nib.load(labelname)
        spacing = label.affine[[0, 1, 2], [0, 1, 2]]
        label = label.get_data()
        label = np.transpose(label, [1, 0, 2])
        ind = ((-spacing > 0) - 0.5) * 2
        label = label[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]
        label = resize(label.astype(np.float64), new_size, anti_aliasing=False, order=0)
        print(imgname, 'loaded', new_size, spacing)

        # get the bounding box of foreground
        # 获取到器官位置 的index
        tempL = np.array(np.where(label > 0))
        print(tempL[0].shape)
        # bbox 每一个维度的最小值和最大值区间
        bbox = np.array([[max(0, np.min(tempL[0]) - pad[0]), min(label.shape[0], np.max(tempL[0]) + pad[0])], \
                         [max(0, np.min(tempL[1]) - pad[1]), min(label.shape[1], np.max(tempL[1]) + pad[1])], \
                         [max(0, np.min(tempL[2]) - pad[2]), min(label.shape[2], np.max(tempL[2]) + pad[2])]])
        # 每一个维度的中心点
        center = np.mean(bbox, 1).astype(int)
        # 最大值 - 最小值获得bbox长度
        bbL = bbox[:, 1] - bbox[:, 0]
        # 取最长的那条边
        L = int(np.max(bbox[:, 1] - bbox[:, 0]))
        print(L)

        # extract a cubic box(立方体) that contain all the foreground
        # 取出label 中PANCREAS的部分
        out = \
            image[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
            max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
            max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]
        Shape = list(out.shape)
        Shape.append(2)
        print(Shape)
        Out_img = out
        Out_label = \
            label[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
            max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
            max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]

        path_prefix = path.join(to_path, path.split(imgname)[1].split(".")[0])
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        np.save(path.join(path_prefix, 'img.npy'), Out_img.astype(np.int16))
        np.save(path.join(path_prefix, 'label.npy'), Out_label.astype(np.int8))
        # np.stack((x,y), axis=-1) 在最后新增一个维度，并在最后一个维度进行拼接， axis默认=0
        np.save(path.join(path_prefix, 'merge.npy'), np.stack((Out_img, Out_label), axis=-1).astype(np.int16))


# create json file
normalfiles = glob("/HDD_data/MING/VAE/PCR_data/OD/normalnii" + '/*.nii.gz')
PCRniifiles = glob('/HDD_data/MING/VAE/PCR_data/OD/PCRnii' + '/*.nii.gz')
normalfiles.sort()
PCRniifiles.sort()
label = {}
mergefiles,_ = file_name("/HDD_data/MING/VAE/PCR_data/OD/npyfile")
root_dir = "/HDD_data/MING/VAE/PCR_data/OD/npyfile"
for file in normalfiles:
    label[re.findall(r'\d+', file)[0]] = 0
for file in PCRniifiles:
    label[re.findall(r'\d+', file)[0]] = 1
pcr_json = {"PCR_train": [], "PCR_valid": []}


for i in range(131):
    # normalname = normalnames[i]
    file = normalfiles[i]
    names = path.split(file)[1].split(".")[0]
    if names in mergefiles:
        if label[re.findall(r'\d+', file)[0]] == 0:
            pcr_json["PCR_train"].append(names + "/merge.npy")

for i in tqdm(range(23)):
    file = PCRniifiles[i]
    names = path.split(file)[1].split(".")[0]
    if names in mergefiles:
        if label[re.findall(r'\d+', file)[0]] == 1:
            pcr_json["PCR_train"].append(names + "/merge.npy")

for i in tqdm(range(131, len(normalfiles))):
    file = normalfiles[i]
    names = path.split(file)[1].split(".")[0]
    if names in mergefiles:
        if label[re.findall(r'\d+', file)[0]] == 0:
            pcr_json["PCR_valid"].append(names + "/merge.npy")

for i in tqdm(range(23, len(PCRniifiles))):
    file = PCRniifiles[i]
    names = path.split(file)[1].split(".")[0]
    if names in mergefiles:
        if label[re.findall(r'\d+', file)[0]] == 1:
            pcr_json["PCR_valid"].append(names + "/merge.npy")

json_str = json.dumps(pcr_json, indent=4)
with open('/home/user02/TUTMING/ming/VAE/data/new_PCR_data.json', 'w') as json_file:
    json_file.write(json_str)

# pcr_json = {"PCR_train": [], "PCR_valid": []}
# fold = 4
# normalnums = 47
# pcrnums = 8
# for j in range(fold):
#     maxnums = (j + 1) * normalnums
#     PCRflag = (j + 1) * pcrnums
#     if (j + 1) == 4:
#         maxpcrnums = 33
#     else:
#         maxpcrnums = (j + 1) * pcrnums
#     if j != 0:
#         for i in range(j * normalnums, maxnums):
#             # normalname = normalnames[i]
#             file = normalfiles[i]
#             names = path.split(file)[1].split(".")[0]
#             if names in mergefiles:
#                 if label[re.findall(r'\d+', file)[0]] == 0:
#                     pcr_json["PCR_train"].append(names + "/merge.npy")
#
#         for i in tqdm(range(j * pcrnums, maxpcrnums)):
#             file = PCRniifiles[i]
#             names = path.split(file)[1].split(".")[0]
#             if names in mergefiles:
#                 if label[re.findall(r'\d+', file)[0]] == 1:
#                     pcr_json["PCR_train"].append(names + "/merge.npy")
#     if j == 0:
#         for i in tqdm(range(j * normalnums, maxnums)):
#             file = normalfiles[i]
#             names = path.split(file)[1].split(".")[0]
#             if names in mergefiles:
#                 if label[re.findall(r'\d+', file)[0]] == 0:
#                     pcr_json["PCR_valid"].append(names + "/merge.npy")
#
#         for i in tqdm(range(j * pcrnums, maxpcrnums)):
#             file = PCRniifiles[i]
#             names = path.split(file)[1].split(".")[0]
#             if names in mergefiles:
#                 if label[re.findall(r'\d+', file)[0]] == 1:
#                     pcr_json["PCR_valid"].append(names + "/merge.npy")
#
# json_str = json.dumps(pcr_json, indent=4)
# with open('/home/user02/TUTMING/ming/VAE/data/fold_{}.json'.format(0), 'w') as json_file:
#     json_file.write(json_str)