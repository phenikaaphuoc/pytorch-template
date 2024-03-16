import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.utils.visualize import Visualizor
from src.utils import logger

class Validation:
    def __init__(self, opt):
        self.device = opt['device']
        self.num_class = opt['model']['arch']['num_class']
        self.visualizor =  Visualizor(opt['val_parameter']["visualize"])
    def validate(self, model, dataloader_list:list,iter = None):
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        count = 0 
        # Use tqdm for the progress bar
        with torch.no_grad():
            for dataloader in dataloader_list:
                count += len(dataloader)
                with tqdm(total=len(dataloader), desc="Validation") as pbar:
                # Iterate through the DataLoader
                    for inputs, targets in dataloader:
                    # Move data to the device
                        model.feedData((inputs,targets))
                        model.inference()
                        outputs = model.get_output()
                        
                    # Append predictions and targets to lists
                        all_preds.extend(outputs)
                        all_targets.extend(targets.cpu().detach().numpy())

                    # Calculate loss
                        total_loss += model.get_current_loss()
                    # Update tqdm
                        pbar.update(1)
                        del inputs
                        del targets
                        

        
        predicted_classes = [np.argmax(pred) for pred in all_preds]
        all_targets = [np.argmax(pred) for pred in all_targets]
        average_loss = total_loss / count
        logger.info(f"iter {iter} have average loss {average_loss} ")
        self.visualizor.visualize(predicted_classes,all_targets,iter)
        model.train()
        
