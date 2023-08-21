# create json
with open("/data/MING/code/VAE_06/data/My_Multi_all.json", 'r') as f:
	json_dict = json.load(f)

trainlist = json_dict.get("NIH_train", [])
vallist = json_dict.get("NIH_val", [])
# NIH
normal_file = "/data/MING/data/NIH/newaug_data/image"
normalROI_file = "/data/MING/data/NIH/newaug_data/label"
# SYN


files = glob(normal_file + '/*.nii.gz')
normalROI_files = glob(normalROI_file + '/*.nii.gz')
files.sort()
normalROI_files.sort()
trainfiles_id = []
for i in range(len(trainlist)):
    trainfiles_id.append(int(re.findall(r"\d+", trainlist[i])[0]))

valfiles_id = []
for i in range(len(vallist)):
    valfiles_id.append(int(re.findall(r"\d+", vallist[i])[0]))
pcr_json = {"NIH_train": [], "NIH_val": []}

for i in tqdm(range(len(trainlist))):
	file = files[i]
	save_file = os.path.split(file)[1]
	if int(re.findall(r"\d+", save_file)[0]) in  trainfiles_id:
		ROIfile = normalROI_files[i]
		ROIname = os.path.split(ROIfile)[1]
		if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
			pcr_json["NIH_train"].append({"image": file, "label": ROIfile})


for i in tqdm(range(len(vallist))):
	file = files[i]
	save_file = os.path.split(file)[1]
	if int(re.findall(r"\d+", save_file)[0]) in  trainfiles_id:
		ROIfile = normalROI_files[i]
		ROIname = os.path.split(ROIfile)[1]
		if re.findall(r"\d+", save_file) == re.findall(r"\d+", ROIname):
			pcr_json["NIH_val"].append({"image": file, "label": ROIfile})
json_str = json.dumps(pcr_json, indent=4)
with open('/data/MING/code/VAE_06/data/swinunter_newaug.json', 'w') as json_file:
	json_file.write(json_str)

# limited data
import json

with open("/data/MING/code/VAE_06/data/My_Multi_all.json", 'r') as f:
	json_dict = json.load(f)
trainlist = json_dict.get("MSD_train", [])
vallist = json_dict.get("MSD_val", [])
trainlist.sort()
vallist.sort()
allfiles = []
for i in range(len(trainlist)):
	allfiles.append(trainlist[i])
for i in range(len(vallist)):
	allfiles.append(vallist[i])

allfiles.sort()
allfiles_id = []
for i in range(len(allfiles)):
	allfiles_id.append(int(re.findall(r"\d+", allfiles[i])[0]))

arrall_id = np.array(allfiles_id)



nih_trainlist = json_dict.get("NIH_train", [])
nih_vallist = json_dict.get("NIH_val", [])
nih_trainlist.sort()
nih_vallist.sort()
nih_allfiles = []
for i in range(len(nih_trainlist)):
	nih_allfiles.append(nih_trainlist[i])
for i in range(len(nih_vallist)):
	nih_allfiles.append(nih_vallist[i])
nih_allfiles = []
for i in range(len(nih_trainlist)):
	nih_allfiles.append(nih_trainlist[i])
for i in range(len(nih_vallist)):
	nih_allfiles.append(nih_vallist[i])

nih_allfiles.sort()
nih_allfiles_id = []
for i in range(len(nih_allfiles)):
	nih_allfiles_id.append(int(re.findall(r"\d+", nih_allfiles[i])[0]))
nih_arrall_id = np.array(nih_allfiles_id)


word_trainlist = json_dict.get("WORD_train", [])
word_vallist = json_dict.get("WORD_val", [])
word_trainlist.sort()
word_vallist.sort()
word_allfiles = []
for i in range(len(word_trainlist)):
	word_allfiles.append(word_trainlist[i])
for i in range(len(word_vallist)):
	word_allfiles.append(word_vallist[i])
word_allfiles = []
for i in range(len(word_trainlist)):
	word_allfiles.append(word_trainlist[i])
for i in range(len(word_vallist)):
	word_allfiles.append(word_vallist[i])

word_allfiles.sort()
word_allfiles_id = []
for i in range(len(word_allfiles)):
	word_allfiles_id.append(int(re.findall(r"\d+", word_allfiles[i])[0]))
word_arrall_id = np.array(word_allfiles_id)

pcr_json = {"MSD_NIHtrain_limited": [], "WORD_val_limited": []}
for i in range(len(msd)):
    flag = int(re.findall(r"\d+", msd[i])[0])
    if flag in allfiles_id:
        index = np.where(arrall_id == flag)[0][0]
        file_path = allfiles[index]
        pcr_json["MSD_NIHtrain_limited"].append(file_path)
for i in range(len(nih)):
    flag = int(re.findall(r"\d+", nih[i])[0])
    if flag in nih_allfiles_id:
        index = np.where(nih_arrall_id == flag)[0][0]
        file_path = nih_allfiles[index]
        pcr_json["MSD_NIHtrain_limited"].append(file_path)
for i in range(len(word)):
    flag = int(re.findall(r"\d+", word[i])[0])
    if flag in word_allfiles_id:
        index = np.where(word_arrall_id == flag)[0][0]
        file_path = word_allfiles[index]
        pcr_json["WORD_val_limited"].append(file_path)
json_str = json.dumps(pcr_json, indent=4)
with open('/data/MING/code/VAE_06/data/limited_data.json', 'w') as json_file:
    json_file.write(json_str)


pcr_json3 = {"NIH_WORDtrain_limited": [], "MSD_val_limited": []}
for i in range(len(msd)):
	flag = int(re.findall(r"\d+", msd[i])[0])
	if flag in allfiles_id:
		index = np.where(arrall_id == flag)[0][0]
		file_path = allfiles[index]
		pcr_json3["MSD_val_limited"].append(file_path)
for i in range(len(nih)):
	flag = int(re.findall(r"\d+", nih[i])[0])
	if flag in nih_allfiles_id:
		index = np.where(nih_arrall_id == flag)[0][0]
		file_path = nih_allfiles[index]
		pcr_json3["NIH_WORDtrain_limited"].append(file_path)
for i in range(len(word)):
	flag = int(re.findall(r"\d+", word[i])[0])
	if flag in word_allfiles_id:
		index = np.where(word_arrall_id == flag)[0][0]
		file_path = word_allfiles[index]
		pcr_json3["NIH_WORDtrain_limited"].append(file_path)
json_str3 = json.dumps(pcr_json3, indent=4)
with open('/data/MING/code/VAE_06/data/NIH_WORD.json', 'w') as json_file:
	json_file.write(json_str3)

pcr_json2 = {"MSD_WORDtrain_limited": [], "NIH_val_limited": []}
for i in range(len(msd)):
	flag = int(re.findall(r"\d+", msd[i])[0])
	if flag in allfiles_id:
		index = np.where(arrall_id == flag)[0][0]
		file_path = allfiles[index]
		pcr_json2["MSD_WORDtrain_limited"].append(file_path)
for i in range(len(nih)):
	flag = int(re.findall(r"\d+", nih[i])[0])
	if flag in nih_allfiles_id:
		index = np.where(nih_arrall_id == flag)[0][0]
		file_path = nih_allfiles[index]
		pcr_json2["NIH_val_limited"].append(file_path)
for i in range(len(word)):
	flag = int(re.findall(r"\d+", word[i])[0])
	if flag in word_allfiles_id:
		index = np.where(word_arrall_id == flag)[0][0]
		file_path = word_allfiles[index]
		pcr_json2["MSD_WORDtrain_limited"].append(file_path)
json_str2 = json.dumps(pcr_json2, indent=4)
with open('/data/MING/code/VAE_06/data/MSD_WORD.json', 'w') as json_file:
	json_file.write(json_str2)