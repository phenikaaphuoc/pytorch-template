import os
import os.path as osp

root_folder = "src"
sub_folders = ["models","utils","configs","archs","pipelines","data"]
file_in_sub_folder = {
    "models":["base_model.py"],
    "utils":["logger.py","registry.py"],
    "configs": ["train_config.yaml","test_config.yaml"],
    "pipelines": ["train_pipeline.py","test_pipeline.py"]
}

for sub_folder in sub_folders:
    sub_folder_path = osp.join(root_folder,sub_folder)
    if osp.exists(sub_folders):
        continue
    os.makedirs(sub_folder_path,exist_ok=True)
    
    #make empty init file each folder
    with open(osp.join(sub_folder_path,"__init__.py"), 'w') as f:
        pass
    try:
        for file in file_in_sub_folder[sub_folder]:
            if osp.exists(osp.join(sub_folder_path,file)):
                continue
            with open(osp.join(sub_folder_path,file), 'w') as f:
                pass
    except :
        pass
if not osp.exists("requirements.txt"):
    with open("requirements.txt","a") as f:
        pass