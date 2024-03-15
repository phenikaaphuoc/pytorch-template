import matplotlib.pyplot as plt 
from src.utils import *
import os 
import numpy as np
import time
from sklearn.metrics import confusion_matrix as CFM,recall_score,precision_score
import pandas as pd
import seaborn as sn
    
def histogram():
    pass
class Visualizor:
    def __init__(self,config):
        self.config = config
        os.makedirs(config["save_folder"],exist_ok=True) 
        with open(config["save_class_name"],"r") as f:
                self.class_name = [class_name.strip() for class_name in f.readlines()]
        self.num_class = len(self.class_name)
    def visualize(self,
                  predict:np.array,
                  groundtruth:np.array ,
                  iter = None):
        
        '''
        predict shape: Batch x N
        groundtryth shape : Batch x N
        path , name , iter just for path image save
        if path is None use nam
        '''
        predict = np.array(predict).reshape(-1)
        groundtruth = np.array(groundtruth).reshape(-1)
        assert(predict.shape == groundtruth.shape)
        id = str(time.time()) if iter is None else "iter_"+str(iter)
        save_folder = os.path.join(self.config["save_folder"],id)
        os.makedirs(save_folder,exist_ok=True)
        self.all_result_file = os.path.join(save_folder,"all_result.txt")
        for metric in self.config["metric"]:
            plt.figure()
            self.save_path = os.path.join(self.config["save_folder"],id,metric+".png")
            getattr(self,metric)(predict,groundtruth)
        
        
    def recall(self,
               predict:np.array,
               groundtruth:np.array):
        recall_value = recall_score(groundtruth,predict,average=None,zero_division=0,labels = range(self.num_class))
        with open(self.all_result_file,"a") as f:
            f.write(f"Mean recall : {np.mean(recall_value)}\n") 
        Visualizor.bar(recall_value,self.save_path,self.class_name)
    def precision(self,
               predict:np.array,
               groundtruth:np.array):
        precision_value = precision_score(groundtruth,predict,average=None,zero_division=1,labels = range(self.num_class))
        with open(self.all_result_file,"a") as f:
            f.write(f"Mean precision : {np.mean(precision_value)}\n")
        Visualizor.bar(precision_value,self.save_path,self.class_name)

        
    def confusion_matrix(self,
                         predict:np.array,
                         groundtruth:np.array):
        confusion_value = CFM(groundtruth,predict,normalize = "true")
        df = pd.DataFrame(confusion_value,index = self.class_name,columns = self.class_name)
        plt.figure(figsize=(len(self.class_name)+3,len(self.class_name)))
        sn.heatmap(df,annot=True)
        plt.savefig(self.save_path)
        logger.info(f"Save a visualize of confusion matrix at {self.save_path}")
        
    def accuracy(self,
                 predict:np.array,
                 groundtruth:np.array):
        accuracy_per_class = CFM(groundtruth,predict,normalize = "true").diagonal()
        Visualizor.bar(accuracy_per_class,self.save_path,self.class_name)
        logger.info(f"Save a visualize of  accuracy at {self.save_path}")
        with open(self.all_result_file,"a") as f:
            f.write(f"Mean accuray : {np.mean(accuracy_per_class)}\n")
        
    @staticmethod
    def bar(value:list,
            save_path,
            x_name : list  = None,
            title :str = None,
            color = "blue"):
        if x_name is None:
            x_name = range(len(value))
        assert(len(x_name)==len(value))
        plt.bar(x_name, value, color=color)
        if title:
            plt.title(title)
        plt.savefig(save_path)

            
        

if __name__ == "__main__":
    predict = [1,2,0,2,1,0]
    target = [1,2,0,1,2,0]
    
