import os
import glob
import pydicom
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return root,files, dirs  # 当前路径下所有非目录子文件,列表


# dcm ---> nii
def dcm2nii_sitk(path_read, path_save, filename):
    """
    path_read: 存放dcm文件的文件夹路径
    path_save: 保存nii文件的文件夹路径
    filename:保存的nii文件名称
    """
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(path_read)
    N = len(seriesIDs)
    lens = np.zeros([N])
    for i in range(N):
        dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[i])
        lens[i] = len(dicom_names)
    N_MAX = np.argmax(lens)
    dicom_names = reader.GetGDCMSeriesFileNames(path_read, seriesIDs[N_MAX])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    sitk.WriteImage(image, path_save+"/" + filename + '.nii.gz')


# def nifti2dicom_1file(nifti_dir, out_dir):
#     """
#     This function is to convert only one nifti file into dicom series
#     `nifti_dir`: the path to the one nifti file
#     `out_dir`: the path to output
#     """
#
#     nifti_file = nib.load(nifti_dir)
#     nifti_array = nifti_file.get_fdata()
#
#     number_slices = nifti_array.shape[2]
#
#     for slice_ in tqdm(range(number_slices)):
#         convertNsave(nifti_array[:,:,slice_], out_dir, slice_)


# png--->nii.gz
def png2nii(png_path, path_save, filename):
    """
    png_path：png文件所在的文件夹
    """
    empt_mat = []
    files=os.listdir(png_path)
    # files.sort(key=lambda x:int(x[9:-4]))
    files.sort()
    for i in files:
        if i!='.DS_Store':
            # print(i)
            img1 = Image.open(os.path.join(png_path, i))
            img2 = np.array(img1, dtype=np.int8)
            # print(img2)
        # 这里取png图片的前三个通道，去除第四个透明通道 方便后续的nii文件的处理
            empt_mat.append(img2)
            # print(empt_mat)
            # print(len(empt_mat))
            emp = np.array(empt_mat)
            # print(emp)
            nii_file = sitk.GetImageFromArray(emp)
            # print(111)
            # print(nii_file)
        # 此处的emp的格式为样本数*高度*宽度*通道数
        # 不要颠倒这些维度的顺序，否则文件保存错误
    nii_path=(path_save+'/' + filename +'.nii.gz')
    sitk.WriteImage(nii_file, nii_path)  # nii_path 为保存路径


# maskMR 操作
# 获取label 非零元素的值
# np.flatnonzero(img)
# img.ravel()[np.flatnonzero(img)]
# or
# img = Image.open("/data1/2021_stu/MING/CHAOS/Train_Sets/MR/10/T2SPIR/Ground/IMG-0043-00021.png")
# img2 = np.array(img, dtype=np.int8)
# flatten
# img2_flatten = img2.flatten()
# to list
# img2_list = flatten_data.tolist()
# to set
# img2_set = set(list_data)
# 查看有几类
def mask_CHAOS_liver(live_path, to_path, filename):
    files = os.listdir(live_path)
    files.sort()
    empt_mat = []
    for i in files:
        img1 = Image.open(os.path.join(live_path, i))
        img2 = np.array(img1, dtype=np.int8)
        # 将其他器官mask 等同于将其他value置0
        # test1 = np.where(img8_test==63, img8_test, 0)
        # img8_test[img8_test!=63] = 0
        img2[img2 != 63] = 0
        img2[img2 == 63] = 1
        empt_mat.append(img2)
        emp = np.array(empt_mat)
        nii_file = sitk.GetImageFromArray(emp)

    nii_path = (to_path+'/' + filename +'.nii.gz')
    sitk.WriteImage(nii_file, nii_path)  # nii_path 为保存路径






if __name__ == "__main__":
    # CHAOS
    # file_dir = "/HDD_data/MING/VAE/CHAOS/Train_Sets/CT"
    # _, files, dirs = file_name(file_dir)
    # CT_data_save = "/HDD_data/MING/VAE/CHAOS/Train_Sets/CT_data"
    # CT_label_save = "/HDD_data/MING/VAE/CHAOS/Train_Sets/CT_label"
    # dirs.sort()
    # for i in tqdm(range(len(dirs))):
    #     dir_name = "Liver_" + str(dirs[i])
    #     label_name = "label" + str(dirs[i])
    #     new_root = os.path.join(file_dir, dirs[i])
    #     _, _, flags = file_name(new_root)
    #     if flags:
    #         for flag in flags:
    #             if flag == "DICOM_anon":
    #                 train_new_root = os.path.join(new_root, flag)
    #                 _, dcms, _ = file_name(train_new_root)
    #                 if dcms:
    #                     dcm_file = train_new_root
    #                 else:
    #                     print("no dcm files")
    #                 dcm2nii_sitk(dcm_file, CT_data_save, dir_name)
    #             elif flag == "Ground":
    #                 label_new_root = os.path.join(new_root, flag)
    #                 _, pngs, _ = file_name(label_new_root)
    #                 if pngs:
    #                     png_file = label_new_root
    #                 else:
    #                     print("no png files")
    #                 png2nii(png_file, CT_label_save, label_name)
    #             else:
    #                 print("No correction dirs")
    # CHAOS MR
    file_dir = "/HDD_data/MING/VAE/CHAOS/Train_Sets/MR"
    _, files, dirs = file_name(file_dir)
    MR_data_save = "/HDD_data/MING/VAE/CHAOS/Train_Sets/MR_data"
    MR_label_save = "/HDD_data/MING/VAE/CHAOS/Train_Sets/MR_label"
    dirs.sort()
    for i in tqdm(range(len(dirs))):
        dir_name = "Liver_" + "T2SPIR" + "_" + str(dirs[i])
        label_name = "label" + "T2SPIR" + "_" + str(dirs[i])
        new_root = os.path.join(file_dir, dirs[i])
        _, _, flags = file_name(new_root)
        if flags:
            for flag in flags:
                if flag == "T1DUAL":
                    print("T1")
                elif flag == "T2SPIR":
                    new_root = os.path.join(new_root, flag)
                    _, _, inners = file_name(new_root)
                    for inner in inners:
                        if inner == "DICOM_anon":
                            train_new_root = os.path.join(new_root, inner)
                            _, dcms, _ = file_name(train_new_root)
                            if dcms:
                                dcm_file = train_new_root
                            else:
                                print("no dcm files")
                            dcm2nii_sitk(dcm_file, MR_data_save, dir_name)
                        elif inner == "Ground":
                            label_new_root = os.path.join(new_root, inner)
                            _, pngs, _ = file_name(label_new_root)
                            if pngs:
                                png_file = label_new_root
                            else:
                                print("no png files")
                            mask_CHAOS_liver(png_file, MR_label_save, label_name)
                        else:
                            print("No correction dirs")
                else:
                    print("error!")



    # NIH
    # file_dir = "/data1/2021_stu/MING/NIH/train/Pancreas-CT"
    # _, files, dirs = file_name(file_dir)
    # save_path = "/data1/2021_stu/MING/NIH/train/data"
    # dirs.sort()
    # for i in tqdm(range(len(dirs))):
    #     new_root = os.path.join(file_dir, dirs[i])
    #     _, _, flags = file_name(new_root)
    #     if flags:
    #         for flag in flags:
    #             new_root = os.path.join(new_root, flag)
    #             _, _, inners = file_name(new_root)
    #             if inners:
    #                 new_root = os.path.join(new_root, inners[0])
    #                 _, dcms, _ = file_name(new_root)
    #                 if dcms:
    #                     dcm_file = new_root
    #                 else:
    #                     print("no dcm files")
    #     dcm2nii_sitk(dcm_file, save_path, dirs[i])

