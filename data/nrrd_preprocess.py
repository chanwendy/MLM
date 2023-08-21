import json
import os
from glob import glob
import vtk
from tqdm import  tqdm
import nrrd
import nibabel as nib
import numpy as np
import re
import nibabel as nib
def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


if __name__ == '__main__':
    # normal
    baseDir = os.path.normpath(r'/HDD_data/MING/VAE/PCR_data/OD/normalOD')
    # PCR
    PCRDir = os.path.normpath("/HDD_data/MING/VAE/PCR_data/OD/PCRcasesOD")
    PCRsave_file = "/HDD_data/MING/VAE/PCR_data/OD/PCRnii"
    PCRROIsave_file = "/HDD_data/MING/VAE/PCR_data/OD/ROIPCR"
    normalsave_file = "/HDD_data/MING/VAE/PCR_data/OD/normalnii"
    normalROIsave_file = "/HDD_data/MING/VAE/PCR_data/OD/ROInormal"
    normalROI_file = "/HDD_data/MING/VAE/PCR_data/ROI"
    PCRROI_file = "/HDD_data/MING/VAE/PCR_data/ROI/PCRcasesROI"
    files = glob(baseDir+'/*.nrrd')
    PCRfiles = glob(PCRDir + '/*.nrrd')
    normalROI_files = glob(normalROI_file+'/*.nrrd')
    PCRROI_files = glob(PCRROI_file+'/*.nrrd')
    files.sort()
    PCRfiles.sort()
    normalROI_files.sort()
    PCRROI_files.sort()
    # try:
    #     for i in tqdm(range(len(files))):
    #         file = files[i]
    #         m, info = readnrrd(file)
    #         save_file = os.path.split(file)[1]
    #         save_file = save_file.replace('.nrrd', '.nii')
    #         save_filename = os.path.join(normalsave_file, save_file)
    #         writenifti(m,  save_filename, info)
    #     for i in tqdm(range(len(PCRfiles))):
    #         file = PCRfiles[i]
    #         m, info = readnrrd(file)
    #         save_file = os.path.split(file)[1]
    #         save_file = save_file.replace('.nrrd', '.nii.gz')
    #         save_filename = os.path.join(PCRsave_file, save_file)
    #         writenifti(m,  save_filename, info)
    # except Exception as e:
    #     print("error {}".format(e))


    # for file in files:
    #     m, info = readnrrd(file)
    #     writenifti(m,  file.replace( '.nrrd','.nii.gz'), info)

    # nrrd 文件保存路径

    # for i in tqdm(range(len(files))):
    #     file = files[i]
    #     data, options = nrrd.read(file)
    #     save_file = os.path.split(file)[1]
    #     save_file = save_file.replace('.nrrd', '.nii.gz')
    #     save_filename = os.path.join(normalROIsave_file, save_file)
    #     img = nib.Nifti1Image(data, np.eye(4))
    #     nib.save(img, save_filename)

    try:
        for i in tqdm(range(len(files))):
            file = files[i]
            ROIfile = normalROI_files[i]
            data, options = nrrd.read(file)
            save_file = os.path.split(file)[1]
            ROIname = os.path.split(ROIfile)[1]
            if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
                ROIdata, ROIoptions = nrrd.read(ROIfile)
                if data.shape[2] != ROIdata.shape[2]:
                    print("shape error")
                    print("id is {}".format(re.findall(r"\d+", save_file)))
                    continue
                savedata = np.concatenate((np.expand_dims(data, axis=3), np.expand_dims(ROIdata, axis=3)), axis=3)
                save_file = save_file.replace('.nrrd', '.nii.gz')
                # save_filename = os.path.join(normalsave_file, save_file)
                save_filename = os.path.join(normalROIsave_file, save_file)
                img = nib.Nifti1Image(savedata, np.eye(4))
                nib.save(img, save_filename)
            else:
                print("error ID is {} {}".format(re.findall(r"\d+", save_file), re.findall(r"\d+", ROIname)))
                savedata = data

        for i in tqdm(range(len(PCRfiles))):
            file = PCRfiles[i]
            ROIfile = PCRROI_files[i]
            data, options = nrrd.read(file)
            save_file = os.path.split(file)[1]
            ROIname = os.path.split(ROIfile)[1]
            if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
                ROIdata, ROIoptions = nrrd.read(ROIfile)
                if data.shape[2] != ROIdata.shape[2]:
                    print("shape error")
                    print("id is {}".format(re.findall(r"\d+", save_file)))
                    continue
                savedata = np.concatenate((np.expand_dims(data, axis=3), np.expand_dims(ROIdata, axis=3)), axis=3)
                save_file = save_file.replace('.nrrd', '.nii.gz')
                # save_filename = os.path.join(PCRsave_file, save_file)
                save_filename = os.path.join(PCRROIsave_file, save_file)
                img = nib.Nifti1Image(savedata, np.eye(4))
                nib.save(img, save_filename)
            else:
                print("error ID is {} {}".format(re.findall(r"\d+", save_file), re.findall(r"\d+", ROIname)))
                savedata = data

    except Exception as e:
        print("error {}".format(e))

    # create json
    normal_file = "/HDD_data/MING/VAE/PCR_data/OD/normalnii"
    PCR_file = "/HDD_data/MING/VAE/PCR_data/OD/PCRnii"
    normalROI_file = "/HDD_data/MING/VAE/PCR_data/OD/normalonlyROI"
    PCRROI_file = "/HDD_data/MING/VAE/PCR_data/OD/PCRonlyROI"
    PCRfiles = glob(PCR_file + '/*.nii.gz')
    files = glob(normal_file + '/*.nii.gz')
    normalROI_files = glob(normalROI_file + '/*.nii.gz')
    PCRROI_files = glob(PCRROI_file + '/*.nii.gz')
    files.sort()
    PCRfiles.sort()
    normalROI_files.sort()
    PCRROI_files.sort()

    pcr_json = {"training": [], "validation": []}
    for i in tqdm(range(131)):
        file = files[i]
        ROIfile = normalROI_files[i]
        save_file = os.path.split(file)[1]
        ROIname = os.path.split(ROIfile)[1]
        if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
            pcr_json["training"].append({"image": file, "label": ROIfile, "cls_label": 0})

    for i in tqdm(range(23)):
        file = PCRfiles[i]
        ROIfile = PCRROI_files[i]
        save_file = os.path.split(file)[1]
        ROIname = os.path.split(ROIfile)[1]
        if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
            pcr_json["training"].append({"image": file, "label": ROIfile, "cls_label": 1})

    for i in tqdm(range(131, len(files))):
        file = files[i]
        ROIfile = normalROI_files[i]
        save_file = os.path.split(file)[1]
        ROIname = os.path.split(ROIfile)[1]
        if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
            pcr_json["validation"].append({"image": file, "label": ROIfile, "cls_label": 0})

    for i in tqdm(range(23, len(PCRfiles))):
        file = PCRfiles[i]
        ROIfile = PCRROI_files[i]
        save_file = os.path.split(file)[1]
        ROIname = os.path.split(ROIfile)[1]
        if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
            pcr_json["validation"].append({"image": file, "label": ROIfile, "cls_label": 1})

    json_str = json.dumps(pcr_json)
    with open('/home/user02/TUTMING/ming/VAE/data/PCR_data.json', 'w') as json_file:
        json_file.write(json_str, )

    # create npy
    pad = [32, 32, 32]
    path_prefix = "/HDD_data/MING/VAE/PCR_data/OD/npyfile/normal"
    for i in tqdm(range(131)):
        file = files[i]
        ROIfile = normalROI_files[i]
        save_file = os.path.split(file)[1]
        ROIname = os.path.split(ROIfile)[1]
        if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
            image = nib.load(file)
            label = nib.load(ROIfile)
            spacing = image.affine[[0, 1, 2], [0, 1, 2]]
            ind = ((-spacing > 0) - 0.5) * 2
            image = image.get_data()
            # 将0 1 维度对换 意义是什么-》旋转90度
            image = np.transpose(image, [1, 0, 2])
            image = image[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]
            spacing = label.affine[[0, 1, 2], [0, 1, 2]]
            label = label.get_data()
            label = np.transpose(label, [1, 0, 2])
            ind = ((-spacing > 0) - 0.5) * 2
            label = label[::int(ind[1]), ::int(ind[0]), ::int(ind[2])]
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
            path_prefix = os.path.join(path_prefix, ROIname)
            if not os.path.exists(path_prefix):
                os.makedirs(path_prefix)

        # for i in range(len(files)):
        #     dcm = pydicom.dcmread(files[i])
        #     dcm.PixelData = labelresize[..., i]
        #     dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        #     dcm.save_as("/HDD_data/MING/VAE/NIH001/{}.dcm".format(i))

    # data_path=r'./data/1.nrrd'
    # save_path='./data/1.nii'
    # data,options=nrrd.read(data_path)  # 读取 nrrd 文件
    # img=nib.Nifti1Image(data,np.eye(4)) # 将 nrrd 文件转换为 .nii 文件
    # nib.save(img,save_path) # 保存 nii 文件
