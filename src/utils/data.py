from src.data import get_dataset
def get_dataloader_helper(dataopt:dict):
    train_data_loader_list  = []
    val_data_loader_list = []

    for data_parse in dataopt.keys():
        data_config_list = dataopt[data_parse]
        for data_config in data_config_list:
            if data_parse == "train":
                train_data_loader_list.append(get_dataset(data_config))
            else: 
                val_data_loader_list.append(get_dataset(data_config))
    train_data_loader_list = train_data_loader_list if train_data_loader_list else None
    val_data_loader_list = val_data_loader_list if val_data_loader_list else None
    return train_data_loader_list,val_data_loader_list