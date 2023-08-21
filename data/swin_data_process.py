import json
import numpy as np
import os.path as path
import nibabel as nib
import os
import glob
# import ipdb
import SimpleITK as sitk
import pydicom
from skimage.transform import resize
from tqdm import tqdm



# image_path = '<path-to-the-image>/nih_data/Pancreas-CT/data' # TODO: modify this.
# label_path = '<path-to-the-data>/nih_data/Pancreas-CT/TCIA_pancreas_labels-02-05-2017' # TODO: modify this.
# to_path = 'data/nih' # TODO: modify this.
# NIH
# image_path = '/data/MING/data/NIH/train/data'
# label_path = '/data/MING/data/NIH/TCIA_pancreas_labels-02-05-2017'
# to_path = '/data/MING/data/NIH/newaug_data/image'  # TODO: modify this.
# label_to_path = "/data/MING/data/NIH/newaug_data/label"
# if not os.path.exists(to_path):
# 	os.makedirs(to_path)
# if not os.path.exists(label_to_path):
# 	os.makedirs(label_to_path)

# synapse
# synapse_path = "/data/MING/data/synapse/RawData/train/images"
# synapse_label_path = "/data/MING/data/synapse/RawData/train/labels"
# synapse_to_path = "/data/MING/data/synapse/newaug_data/image"
# synapse_label_to_path = "/data/MING/data/synapse/newaug_data/label"
# if not os.path.exists(synapse_to_path):
# 	os.makedirs(synapse_to_path)
# if not os.path.exists(synapse_label_to_path):
# 	os.makedirs(synapse_label_to_path)
# WORD
# word_path = "/data/MING/data/WORD/images"
# word_label_path = "/data/MING/data/WORD/labels"
# word_path = "/data/MING/data/WORD/imagesVal"
# word_label_path = "/data/MING/data/WORD/labelsVal"
# bladder
# word_to_path = "/data/MING/data/WORD/bladder_newaug_data/image"
# word_label_to_path = "/data/MING/data/WORD/bladder_newaug_data/label"
# spleen
# word_to_path = "/data/MING/data/WORD/spleen_newaug_data/image"
# word_label_to_path = "/data/MING/data/WORD/spleen_newaug_data/label"
# word_to_path = "/data/MING/data/WORD/lkidney_newaug_data/image"
# word_label_to_path = "/data/MING/data/WORD/lkidney_newaug_data/label"
# for sifa
# word_to_path = "/data/MING/data/SIFA/npz_data/WORD/images_data"
# word_label_to_path = "/data/MING/data/SIFA/npz_data/WORD/labels_data"
# if not os.path.exists(word_to_path):
# 	os.makedirs(word_to_path)
# if not os.path.exists(word_label_to_path):
# 	os.makedirs(word_label_to_path)

# # MSD
# MSD_path = "/data/MING/data/MSD/Task07_Pancreas/imagesTr"
# MSD_label_path = "/data/MING/data/MSD/Task07_Pancreas/labelsTr"
# msd_to_path = "/data/MING/data/MSD/Task07_Pancreas/newaug_data/image"
# msd_label_to_path = "/data/MING/data/MSD/Task07_Pancreas/newaug_data/label"
# if not os.path.exists(msd_to_path):
# 	os.makedirs(msd_to_path)
# if not os.path.exists(msd_label_to_path):
# 	os.makedirs(msd_label_to_path)

# CHAOS
# names = glob.glob(path.join(CHAOS_path, '*.gz'))
# synapse
# names = glob.glob(path.join(synapse_path, '*.gz'))
# MSD
# names = glob.glob(path.join(MSD_path, '*.gz'))
# WORD
names = glob.glob(path.join(word_path, '*.gz'))
# NIH
# names = glob.glob(path.join(image_path, '*.gz')) 	# ? NIH 不是dcm？
# ipdb.set_trace()
names.sort()
# path split 将最后一个文件和前面路径分开  ('/data1/2021_stu/MING/NIH/train/data', 'PANCREAS_0001.nii.gz')
names = [path.split(f)[1] for f in names]
# 大概的文件格式应该是root ->Pancreas-CT/data -> PANCREAS_0002_1-001.nii.gz

pad = [32,32,32]
# img_name is PANCREAS_0002
for i in tqdm(range(len(names))):
# for img_name in names:
	# 0001 0002
	img_name = names[i]
	# NIH
	# label_name = 'label' + img_name.split('_')[1]  # TODO: modify this.
	# label_name = 'label' + img_name.split('.')[0][3:7] # for synapse
	# CHAOS
	# label_name = "label" + img_name.split('_')[1]
	# CHAOS MR
	# label_name = "label" + img_name.split('_')[1] + "_" + img_name.split('_')[2]
	# MSD WORD
	label_name = img_name

	# NIH
	# image = nib.load(path.join(image_path, img_name))
	# CHAOS
	# image = nib.load(path.join(CHAOS_path, img_name))
	# synapse
	# image = nib.load(path.join(synapse_path, img_name))
	# MSD
	# image = nib.load(path.join(MSD_path, img_name))
	# word
	image = nib.load(path.join(word_path, img_name))
	spacing = image.affine[[0,1,2], [0,1,2]]


	# deciding the direction
	# 将spacing 值转为1或-1，
	ind = ((-spacing>0)-0.5)*2
	image = image.get_fdata()
	# 将0 1 维度对换 意义是什么-》旋转90度
	image = np.transpose(image,[1,0,2])
	image = image[::int(ind[1]),::int(ind[0]),::int(ind[2])]
	aug_spacing = np.array([1.5, 1.5, 2.0])
	# resample to 1mm
	new_size = (np.array(image.shape)*np.abs(spacing)).astype(np.int)
	new_size = (new_size / np.abs(aug_spacing)).astype(np.int)

	image = resize(image.astype(np.float64), new_size)
	# NIH
	# label = nib.load(path.join(label_path, label_name))
	# CHAOS
	# label = nib.load(path.join(CHAOS_label_path, label_name))
	# synapse
	# label = nib.load(path.join(synapse_label_path, label_name + ".nii.gz"))
	# MSD
	# label = nib.load(path.join(MSD_label_path, label_name))
	# word
	label = nib.load(path.join(word_label_path, label_name))
	spacing = label.affine[[0,1,2],[0,1,2]]
	label = label.get_fdata()

	label[label != 8] = 0
	label[label == 8] = 1

	label = np.transpose(label,[1,0,2])
	ind = ((-spacing>0)-0.5)*2
	label = label[::int(ind[1]),::int(ind[0]),::int(ind[2])]
	label = resize(label.astype(np.float64),new_size,anti_aliasing=False,order=0)
	print(img_name, 'loaded', new_size, spacing)

	# get the bounding box of foreground
	# 获取到器官位置 的index
	tempL = np.array(np.where(label > 0))
	bbox = np.array([[max(0, np.min(tempL[0]) - pad[0]), min(label.shape[0], np.max(tempL[0]) + pad[0])], \
					 [max(0, np.min(tempL[1]) - pad[1]), min(label.shape[1], np.max(tempL[1]) + pad[1])], \
					 [max(0, np.min(tempL[2]) - pad[2]), min(label.shape[2], np.max(tempL[2]) + pad[2])]])
	# 每一个维度的中心点
	center = np.mean(bbox, 1).astype(int)
	# 最大值 - 最小值获得bbox长度
	bbL = bbox[:, 1] - bbox[:, 0]
	# 取最长的那条边
	L = int(np.max(bbox[:, 1] - bbox[:, 0]))
	crop_index1 = np.array([[max(0, center[0] - int(L / 2)), min(label.shape[0], center[0] - int(L / 2) + L)],
							[max(0, center[1] - int(L / 2)), min(label.shape[1], center[1] - int(L / 2) + L)],
							[max(0, center[2] - int(L / 2)), min(label.shape[2], center[2] - int(L / 2) + L)]
							])

	crop_img1 = \
		image[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
		max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
		max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]
	crop_label1 = \
		label[max(0, center[0] - int(L / 2)):min(label.shape[0], center[0] - int(L / 2) + L), \
		max(0, center[1] - int(L / 2)):min(label.shape[1], center[1] - int(L / 2) + L), \
		max(0, center[2] - int(L / 2)):min(label.shape[2], center[2] - int(L / 2) + L)]

	index = np.array(np.where(crop_label1 > 0)).T
	shift = 0
	if index.shape[0] > 0:
		# 取第0维的最大值和最小值
		bbox_max = np.max(index, 0)
		bbox_min = np.min(index, 0)
		center = (bbox_max + bbox_min) // 2
		L = np.max(bbox_max - bbox_min)
		pad_width = int(L * 0.1)
	else:
		center = np.array([64, 64, 64])
		L = 32
		pad_width = int(L * 0.1)
	crop_index2 = np.array([[max(center[0] - L // 2 - pad_width + shift, 0),
							 min(center[0] + L // 2 + pad_width + shift, crop_label1.shape[0])],
							[max(center[1] - L // 2 - pad_width + shift, 0),
							 min(center[1] + L // 2 + pad_width + shift, crop_label1.shape[1])],
							[max(center[2] - L // 2 - pad_width + shift, 0),
							 min(center[2] + L // 2 + pad_width + shift, crop_label1.shape[2])]
							])
	crop_label2 = crop_label1[
				  max(center[0] - L // 2 - pad_width + shift, 0):min(center[0] + L // 2 + pad_width + shift,
																	 crop_label1.shape[0]), \
				  max(center[1] - L // 2 - pad_width + shift, 0):min(center[1] + L // 2 + pad_width + shift,
																	 crop_label1.shape[1]), \
				  max(center[2] - L // 2 - pad_width + shift, 0):min(center[2] + L // 2 + pad_width + shift,
																	 crop_label1.shape[2])]
	diff = list(L + pad_width * 2 - np.array(crop_label2.shape))
	axis_pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
	pad_label = np.pad(crop_label2, axis_pad_width)
	crop_img2 = crop_img1[max(center[0] - L // 2 - pad_width + shift, 0):min(center[0] + L // 2 + pad_width + shift,
																			 crop_img1.shape[0]), \
				max(center[1] - L // 2 - pad_width + shift, 0):min(center[1] + L // 2 + pad_width + shift,
																   crop_img1.shape[1]), \
				max(center[2] - L // 2 - pad_width + shift, 0):min(center[2] + L // 2 + pad_width + shift,
																   crop_img1.shape[2])]
	diff = list(L + pad_width * 2 - np.array(crop_img2.shape))
	axis_pad_width = [(int(cur_diff / 2), cur_diff - int(cur_diff / 2)) for cur_diff in diff]
	pad_img = np.pad(crop_img2, axis_pad_width)
	# 7. resize 到128
	Out_img = resize(pad_img, (128, 128, 128))
	Out_label = resize(pad_label, (128, 128, 128))


	# /data1/2021_stu/MING/NIH/  PANCREAS_0002
	# NIH
	# path_prefix = path.join(CHAOS_to_path, img_name.split('.')[0])
	# CHAOS
	# path_prefix = path.join(CHAOS_to_path, img_name.split('.')[0])
	# synapse
	# path_prefix = path.join(synapse_to_path, img_name.split('.')[0]) + ".nii.gz"
	# label_path_prefix = path.join(synapse_label_to_path, img_name.split('.')[0]) + ".nii.gz"
	# WORD
	path_prefix = path.join(word_to_path, img_name.split('.')[0]) + ".nii.gz"
	label_path_prefix = path.join(word_label_to_path, img_name.split('.')[0]) + ".nii.gz"
	# MSD
	# path_prefix = path.join(MSD_to_path, img_name.split('.')[0])
	# path_prefix = path.join(msd_to_path, img_name.split('.')[0]) + ".nii.gz"
	# label_path_prefix = path.join(msd_label_to_path, img_name.split('.')[0]) + ".nii.gz"
	# NIH
	# path_prefix = path.join(to_path, img_name.split('.')[0])+ ".nii.gz"
	# label_path_prefix = path.join(label_to_path, img_name.split('.')[0]) + ".nii.gz"

	# 每一个image或label 维度为（H, W, slice）
	nii_file = np.swapaxes(Out_img, 0, 2)
	# nii_file = np.swapaxes(nii_file, 1, 2)
	# nii_file = nii_file[::-1, ::1, ::1]
	nii_file = sitk.GetImageFromArray(nii_file)
	sitk.WriteImage(nii_file, path_prefix)

	nii_file = np.swapaxes(Out_label, 0, 2)
	# 不做这两部的话其他两个视图就不太对， 做了这两步的话，重新导入进来又不太对
	# nii_file = np.swapaxes(nii_file, 1, 2)
	# nii_file = nii_file[::-1, ::1, ::1]
	nii_file = sitk.GetImageFromArray(nii_file)
	sitk.WriteImage(nii_file, label_path_prefix)


