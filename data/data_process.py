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



# image_path = '<path-to-the-image>/nih_data/Pancreas-CT/data' # TODO: modify this.
# label_path = '<path-to-the-data>/nih_data/Pancreas-CT/TCIA_pancreas_labels-02-05-2017' # TODO: modify this.
# to_path = 'data/nih' # TODO: modify this.
# NIH
# image_path = '/data1/2021_stu/MING/NIH/train/data'
# label_path = '/data1/2021_stu/MING/NIH/TCIA_pancreas_labels-02-05-2017'
# to_path = '/data1/2021_stu/MING/NIH'  # TODO: modify this.
# if not os.path.exists(to_path):
# 	os.makedirs(to_path)
# CHAOS CT
# CHAOS_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/CT_data"
# CHAOS_label_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/CT_label"
# CHAOS_to_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/data"
# if not os.path.exists(CHAOS_to_path):
# 	os.makedirs(CHAOS_to_path)
# CHAOS_MR
# CHAOS_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/MR_data"
# CHAOS_label_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/MR_label"
# CHAOS_to_path = "/HDD_data/MING/VAE/CHAOS/Train_Sets/mr_data"
# if not os.path.exists(CHAOS_to_path):
# 	os.makedirs(CHAOS_to_path)
# synapse
synapse_path = "/HDD_data/MING/VAE/synapse/RawData/train/images"
synapse_label_path = "/HDD_data/MING/VAE/synapse/RawData/train/labels"
synapse_to_path = "/HDD_data/MING/VAE/synapse/cube"
if not os.path.exists(synapse_to_path):
	os.makedirs(synapse_to_path)

# # MSD
# MSD_path = "/data1/2021_stu/MING/MSD/Task07_Pancreas/imagesTr"
# MSD_label_path = "/data1/2021_stu/MING/MSD/Task07_Pancreas/labelsTr"
# MSD_to_path = "/data1/2021_stu/MING/MSD/Task07_Pancreas/data"


# CHAOS
# names = glob.glob(path.join(CHAOS_path, '*.gz'))
# synapse
names = glob.glob(path.join(synapse_path, '*.gz'))
# MSD
# names = glob.glob(path.join(MSD_path, '*.gz'))
# NIH
# names = glob.glob(path.join(image_path, '*.dcm')) 	# ? NIH 不是dcm？
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
	label_name = 'label' + img_name.split('.')[0][3:7] # for synapse
	# CHAOS
	# label_name = "label" + img_name.split('_')[1]
	# CHAOS MR
	# label_name = "label" + img_name.split('_')[1] + "_" + img_name.split('_')[2]
	# MSD
	# label_name = img_name

	# NIH
	# image = nib.load(path.join(image_path, img_name))
	# CHAOS
	# image = nib.load(path.join(CHAOS_path, img_name))
	# synapse
	image = nib.load(path.join(synapse_path, img_name))
	# MSD
	# image = nib.load(path.join(MSD_path, img_name))
	spacing = image.affine[[0,1,2], [0,1,2]]
    
	# deciding the direction
	# 将spacing 值转为1或-1，
	ind = ((-spacing>0)-0.5)*2
	image = image.get_data()
	# 将0 1 维度对换 意义是什么-》旋转90度
	image = np.transpose(image,[1,0,2])
	image = image[::int(ind[1]),::int(ind[0]),::int(ind[2])]
    
	# resample to 1mm
	new_size = (np.array(image.shape)*np.abs(spacing)).astype(np.int)
	image = resize(image.astype(np.float64), new_size)
	# NIH
	# label = nib.load(path.join(label_path, label_name))
	# CHAOS
	# label = nib.load(path.join(CHAOS_label_path, label_name))
	# synapse
	label = nib.load(path.join(synapse_label_path, label_name + ".nii.gz"))
	# MSD
	# label = nib.load(path.join(MSD_label_path, label_name))
	spacing = label.affine[[0,1,2],[0,1,2]]
	label = label.get_data()
	label = np.transpose(label,[1,0,2])
	ind = ((-spacing>0)-0.5)*2
	label = label[::int(ind[1]),::int(ind[0]),::int(ind[2])]
	label = resize(label.astype(np.float64),new_size,anti_aliasing=False,order=0)
	print(img_name, 'loaded', new_size, spacing)
    
	# get the bounding box of foreground
	# 获取到器官位置 的index
	tempL = np.array(np.where(label>0))
	print(tempL[0].shape)
	# bbox 每一个维度的最小值和最大值区间
	bbox = np.array([[max(0, np.min(tempL[0])-pad[0]), min(label.shape[0], np.max(tempL[0])+pad[0])], \
	[max(0, np.min(tempL[1])-pad[1]), min(label.shape[1], np.max(tempL[1])+pad[1])], \
	[max(0, np.min(tempL[2])-pad[2]), min(label.shape[2], np.max(tempL[2])+pad[2])]])
	# 每一个维度的中心点
	center = np.mean(bbox,1).astype(int)
	# 最大值 - 最小值获得bbox长度
	bbL = bbox[:,1]-bbox[:,0]
	# 取最长的那条边
	L = int(np.max(bbox[:,1]-bbox[:,0]))
	print(L)

	# extract a cubic box(立方体) that contain all the foreground
	# 取出label 中PANCREAS的部分
	out = \
		image[max(0,center[0]-int(L/2)):min(label.shape[0],center[0]-int(L/2)+L),\
		max(0,center[1]-int(L/2)):min(label.shape[1],center[1]-int(L/2)+L),\
		max(0,center[2]-int(L/2)):min(label.shape[2],center[2]-int(L/2)+L)]
	Shape = list(out.shape)
	Shape.append(2)
	print(Shape)
	Out_img = out
	Out_label = \
	label[max(0,center[0]-int(L/2)):min(label.shape[0],center[0]-int(L/2)+L),\
	max(0,center[1]-int(L/2)):min(label.shape[1],center[1]-int(L/2)+L),\
	max(0,center[2]-int(L/2)):min(label.shape[2],center[2]-int(L/2)+L)]
	# /data1/2021_stu/MING/NIH/  PANCREAS_0002
	# NIH
	# path_prefix = path.join(CHAOS_to_path, img_name.split('.')[0])
	# CHAOS
	# path_prefix = path.join(CHAOS_to_path, img_name.split('.')[0])
	# synapse
	path_prefix = path.join(synapse_to_path, img_name.split('.')[0])
	# MSD
	# path_prefix = path.join(MSD_to_path, img_name.split('.')[0])
	if not os.path.exists(path_prefix):
		os.makedirs(path_prefix)
	# 每一个image或label 维度为（H, W, slice）
	np.save(path.join(path_prefix, 'img.npy'), Out_img.astype(np.int16))
	np.save(path.join(path_prefix, 'label.npy'), Out_label.astype(np.int8))
	# np.stack((x,y), axis=-1) 在最后新增一个维度，并在最后一个维度进行拼接， axis默认=0
	np.save(path.join(path_prefix, 'merge.npy'), np.stack((Out_img,Out_label),axis=-1).astype(np.int16))

