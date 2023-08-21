import json
import os
import glob
import re
import shutil
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

# 1.从我的json文件中获取到训练文件和验证集文件的id，同时获得对应的label
# todo:modify json_path train_key test_key des_root
json_path = "/home/2021_stu/TUTMING/VAE/data/My_Multi_all.json"
# NIH
# train_key = "NIH_train"
# test_key = "NIH_val"
# MSD
# train_key = "MSD_train"
# test_key = "MSD_val"
# SYN
# train_key = "SYN_train"
# test_key = "SYN_val"
# word
# train_key = "WORD_train"
# test_key = "WORD_val"
# AMOS
train_key = "AMOS_train_mr"
test_key = "AMOS_val_mr"
# NIH
# des_root = '/data2/2021_stu/MING/nnUNet_raw_data_base/nnUNet_raw_data/Task502_NIHpan'
# MSD
# des_root = '/data2/2021_stu/MING/nnUNet_raw_data_base/nnUNet_raw_data/Task503_MSDpan'
# SYN
# des_root = '/data2/2021_stu/MING/nnUNet_raw_data_base/nnUNet_raw_data/Task501_SYNpan'
# WORD
# des_root = "/data2/2021_stu/MING/nnUNet_raw_data_base/nnUNet_raw_data/Task504_WORDpan"
# AMOS
des_root = "/data2/2021_stu/MING/nnUNet_raw_data_base/nnUNet_raw_data/Task505_AMOSpan"

des_train = os.path.join(des_root, "imagesTr")
if not os.path.exists(des_train):
    os.mkdir(des_train)
des_train_label = os.path.join(des_root, "labelsTr")
if not os.path.exists(des_train_label):
    os.mkdir(des_train_label)
des_test = os.path.join(des_root, "imagesTs")
if not os.path.exists(des_test):
    os.mkdir(des_test)
des_test_label = os.path.join(des_root, "labelsTs")
if not os.path.exists(des_test_label):
    os.mkdir(des_test_label)

# nih 文件路径
# todo: modify  nih_path nih_label_path
# NIH
# nih_path = "/data1/2021_stu/MING/NIH/train/data"
# nih_label_path = "/data1/2021_stu/MING/NIH/TCIA_pancreas_labels-02-05-2017"
# MSD
# msd_path = "/data1/2021_stu/MING/MSD/Task07_Pancreas/imagesTr"
# msd_label_path = "/data1/2021_stu/MING/MSD/Task07_Pancreas/labelsTr"
# SYN
# syn_path = "/data1/2021_stu/MING/synapse/RawData/train/images"
# syn_label_path = "/data1/2021_stu/MING/synapse/RawData/train/labels"
# WORD
# word_path = "/data1/2021_stu/MING/WORD/images"
# word_label_path = "/data1/2021_stu/MING/WORD/labels"
# AMOS
amos_path = "/data1/2021_stu/MING/AMOS/images"
amos_label_path = "/data1/2021_stu/MING/AMOS/labels"

def filedict_from_json_fornn(json_path, key):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])
    return listdict
# 2.将对应id 移动到nnunet的数据集文件夹中（我们的验证集作为测试集）
nih_train_list = filedict_from_json_fornn(json_path, train_key)
train_list_id = []
for i in range(len(nih_train_list)):
    pid = os.path.split(nih_train_list[i])[0]
    id = re.findall(r'\d+', pid)[0]
    train_file = os.path.join(amos_path, pid+".nii.gz")
    # for NIH and syn
    # label_file = os.path.join(syn_label_path, "label" + id+".nii.gz")
    # for other
    label_file = os.path.join(amos_label_path, pid+".nii.gz")
    # for syn
    # shutil.copy(train_file, os.path.join(des_train, "pancreas_" + str(id) + "_0000.nii.gz"))
    # shutil.copy(label_file, os.path.join(des_train_label,"pancreas_" + str(id) + ".nii.gz"))
    # for other
    shutil.copy(train_file, os.path.join(des_train, pid + "_0000.nii.gz"))
    shutil.copy(label_file, os.path.join(des_train_label, pid + ".nii.gz"))

    train_list_id.append(id)

nih_test_list = filedict_from_json_fornn(json_path, test_key)
test_list_id = []
for i in range(len(nih_test_list)):
    pid = os.path.split(nih_test_list[i])[0]
    id = re.findall(r'\d+', pid)[0]
    test_file = os.path.join(amos_path, pid + ".nii.gz")
    # for other
    test_label_file = os.path.join(amos_label_path, pid+".nii.gz")
    # for syn and nih
    # test_label_file = os.path.join(syn_label_path, "label" + id + ".nii.gz")
    # for syn
    # shutil.copy(test_file, os.path.join(des_test, "pancreas_" + str(id) + "_0000.nii.gz"))
    # shutil.copy(test_label_file, os.path.join(des_test_label, "pancreas_" + str(id) + ".nii.gz"))
    # for other
    shutil.copy(test_file, os.path.join(des_test, pid + "_0000.nii.gz"))
    shutil.copy(test_label_file, os.path.join(des_test_label, pid + ".nii.gz"))

    test_list_id.append(id)
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return dirs,files  # 当前路径下所有非目录子文件,列表
# 3.生成nnunet的dataset.json文件
json_dict = OrderedDict()
# todo:modoify name
json_dict['name'] = "AMOS_PAN"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "3D"
json_dict['reference'] = "nothing"
json_dict['licence'] = "nothing"
json_dict['release'] = "0.0"

# 不同模态对应的ID
json_dict['modality'] = {
    "0": "CT",
}
# 不同标签对应的one-hot码
json_dict['labels'] = {
    "0": "background",
    "10": "pancreas",
}

json_dict['numTraining'] = len(nih_train_list)
json_dict['numTest'] = len(nih_test_list)
# for syn
# json_dict['training'] = []
# json_dict['test'] = []
# for i in range(len(nih_train_list)):
#     pid = os.path.split(nih_train_list[i])[0]
#     id = re.findall(r'\d+', pid)[0]
#     json_dict['training'].append({'image': "./imagesTr/{}.nii.gz".format("pancreas_" + str(id)), "label": "./labelsTr/{}.nii.gz".format("pancreas_" + str(id))})
#
# for i in range(len(nih_test_list)):
#     pid = os.path.split(nih_test_list[i])[0]
#     id = re.findall(r'\d+', pid)[0]
#     json_dict['test'].append("./imagesTs/{}.nii.gz".format("pancreas_" + str(id)))

# for other
json_dict['training'] = [{'image': "./imagesTr/{}.nii.gz".format(file[i].split("_0000")[0]), "label": "./labelsTr/{}".format(labelfile[i])} for i in
                         range(len(labelfile))]
json_dict['test'] = ["./imagesTs/{}.nii.gz".format(testfile[i].split("_0000")[0]) for i in
                         range(len(testfile))]
# 将字典写入json文件中
save_json(json_dict, join("/data/MING/data/2021flare/nnUNet_raw_data_base/nnUNet_raw_data/Task001_21FLARE", "dataset.json"))

