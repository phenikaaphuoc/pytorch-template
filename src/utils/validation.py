import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from src.utils.visualize import Visualizor


class Validation:
    def __init__(self, opt):
        self.device = opt['device']
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
                        model.feed_data(inputs,targets)
                        outputs = model().get_output()

                    # Assuming softmax activation for multi-class classification
                        predictions = F.softmax(outputs, dim=1)

                    # Append predictions and targets to lists
                        all_preds.extend(predictions.cpu().detach().numpy())
                        all_targets.extend(targets.cpu().detach().numpy())

                    # Calculate loss
                        loss = model.loss_fn(outputs, targets)
                        total_loss += loss.item()
                    # Update tqdm
                        pbar.update(1)
                        del inputs
                        del targets
                        import pdb;pdb.set_trace()
        
        predicted_classes = [np.argmax(pred) for pred in all_preds]
        all_targets = [np.argmax(pred) for pred in all_targets]
        average_loss = total_loss / count

        model.train()
        return  average_loss
