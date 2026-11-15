import os
os.environ["MPLBACKEND"] = "Agg"   # must be set before importing matplotlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm 
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
from utils import get_args_train
from torch.optim.lr_scheduler import ReduceLROnPlateau



from utils_models import FocalLoss


#################################
############
### Configuration #####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('=====>', device)

#num_classes = 2
#image_size = 392
#batch_size = 3
#num_epochs = 20
#learning_rate = 0.0005
#csv_path = "./data.csv"
#model_name = 'ViT'
#model_name = 'convnext'

args = get_args_train()
print("====> Arguments:", args)

num_classes = args.num_classes
image_size = args.image_size
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
csv_path = args.csv_path

patience_epochs_LRplateau = args.patience_epochs_LRplateau
patience_epochs_early_stop = args.patience_epochs_early_stop

train_strategy = args.train_strategy
num_ensemble_models = args.num_ensemble_models 

model_name = args.model_name
experiment_name = args.experiment_name
model_save_path_parent = os.path.join("./trainedModels", experiment_name)
os.makedirs(model_save_path_parent, exist_ok=True)
model_save_path = os.path.join(model_save_path_parent, model_name + "_model.pth")


#################################
############
### Model Initialization #######
from utils_models import train_model 
from utils_models import get_model
model = get_model(model_name,image_size, num_classes, verbose=True)
model = model.to(device)

#################################
############
### get dataset, transforms ######
from utils_data import RetinalDataset, prepare_data, get_transforms
train_transforms, val_test_transforms, inference_transform = get_transforms(image_size)


def run_one_training(model_save_path):

    train_df, val_df, class_counts, class_weights = prepare_data(csv_path, val_size=0.25)
    train_dataset = RetinalDataset(train_df,  transform=train_transforms)
    val_dataset = RetinalDataset(val_df,  transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True , drop_last=True)

    
    # Loss 
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("=====> Using Weighted CrossEntropyLoss")
    elif args.loss_type == "focal":
        criterion = FocalLoss(
            weight=class_weights.to(device),
            gamma=args.focal_gamma,          # used if schedule=="fixed"
            reduction="mean",
            schedule=args.focal_schedule,    # "fixed" or "flsd53"
        )
        print("=====> Using FocalLoss")
        print("         Focal Loss gamma:", args.focal_gamma)
        print("         Focal Loss schedule:", args.focal_schedule)
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # 'min' for val_loss, 'max' if you track val_acc
        factor=0.5,        # multiply LR by this factor
        patience=patience_epochs_LRplateau,        # epochs with no improvement before reducing
        threshold=1e-4,    # minimal change to qualify as improvement
        cooldown=0,        # epochs to wait after LR reduction
        min_lr=1e-7,       # floor on LR
        #verbose=True
    )

    train_model(model, num_epochs, 
                train_loader, val_loader, 
                device, 
                criterion, 
                optimizer,
                scheduler, 
                patience_epochs_early_stop,
                model_save_path)

#################################
############
##### Run Training ######
if __name__ == "__main__":   
    
    if train_strategy == "none":
        run_one_training(model_save_path)

    if train_strategy == "deepensemble":
        # For deep ensemble, you would typically run multiple training sessions

        for seed in range(num_ensemble_models):
            print("=====>", f"Training model {seed + 1}/{num_ensemble_models} with seed {seed}")
            model = get_model(model_name,image_size, num_classes)
            model = model.to(device)
            model_save_path = os.path.join(model_save_path_parent, model_name + "_model_"+str(seed)+".pth")
            run_one_training(model_save_path)            
       
