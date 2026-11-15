############ define model architecture ############
import timm 
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg", force=True)   # ensure non-interactive backend
import matplotlib.pyplot as plt
import time
from timm.data import resolve_data_config, create_transform
import pickle
import torch.nn as nn

from pathlib import Path
import requests


########################################
################################################################################
########################################
##### ############# Focal Loss ############

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=3.0, reduction="mean", schedule="fixed"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.schedule = schedule
        # we keep CE as base loss
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        """
        logits: [B, K]
        targets: [B]
        """
        # per-sample CE
        ce_loss = self.ce(logits, targets)  # [B]

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)              # [B, K]
            pt = probs[torch.arange(logits.size(0)), targets]  # [B]

            if self.schedule == "fixed":
                gamma_t = torch.full_like(pt, self.gamma)      # [B]
            elif self.schedule == "flsd53":
                # FLSD-53: gamma = 5 for pt in [0,0.2), gamma = 3 otherwise
                gamma_t = torch.where(pt < 0.2,
                                      torch.tensor(5.0, device=pt.device, dtype=pt.dtype),
                                      torch.tensor(3.0, device=pt.device, dtype=pt.dtype))
            else:
                raise ValueError(f"Unknown focal schedule: {self.schedule}")

        focal_factor = (1.0 - pt) ** gamma_t                  # [B]
        loss = focal_factor * ce_loss                         # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


########################################
################################################################################
########################################
##### ############# Get Models ############

def get_model(model_name, image_size, num_classes, verbose=False):
    if model_name == 'ViT':
        return ViT_arch(image_size, num_classes, verbose)
    if model_name == 'convnext':
        return convnext_arch(image_size, num_classes, verbose)
    if model_name == 'RETFoundGreen':
        return RETFoundGreen_arch(image_size, num_classes, verbose)
    


########################################
################################################################################
########################################
##### ############# Models ############

def RETFoundGreen_arch(image_size, num_classes, verbose=False):
    # Load RETFound Green Feature extractor
    rfg = timm.create_model(
        'vit_small_patch14_reg4_dinov2',
                img_size=(image_size, image_size), 
                num_classes=0, 
                drop_rate=0.1, 
                )

    URL = "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth"
    DEST = Path("./trainedModels/retfoundgreen_statedict.pth")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(DEST, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"Saved to {DEST.resolve()}")
    else:
        print(f"Already exists: {DEST.resolve()}")
        rfg_weights = torch.load('./trainedModels/retfoundgreen_statedict.pth')

    rfg.load_state_dict(rfg_weights)
    
    # Check feature dim
    feature_dim = rfg.num_features
    
    # Add custom head
    rfg.head = nn.Sequential(
        # nn.Linear(feature_dim, 512),
        # nn.BatchNorm1d(512),
        # nn.ReLU(),
        # nn.Dropout(0.4),
        # nn.Linear(512, num_classes)
        nn.Linear(feature_dim, num_classes),
    )
    
    # Step 3: Freeze all parameters first
    for param in rfg.parameters():
        param.requires_grad = False

    # Step 4: Unfreeze the classifier parameters
    for param in rfg.head.parameters():
        param.requires_grad = True

    if verbose:
        print('======' * 10)
        print(f"\nTotal parameters: {sum(p.numel() for p in rfg.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in rfg.parameters() if p.requires_grad)}")
        print('======' * 10)
        print('======' * 10)

    return rfg


########################################
########################################



def ViT_arch(image_size, num_classes, verbose=False): 

    model = timm.create_model(
        'vit_small_patch14_reg4_dinov2',
        pretrained=True, img_size=(image_size, image_size),
        num_classes=num_classes, # Two logits for binary classification with CrossEntropyLoss
        drop_rate=0.1, # embedding & MLP dropout
        #attn_drop_rate=0.1,   # attention dropout
        #drop_path_rate=0.1    # stochastic depth
    )

    # Step 3: Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Step 4: Unfreeze the classifier parameters
    for param in model.head.parameters():
        param.requires_grad = True


    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print('======' * 10)
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Frozen parameters: {total_params - trainable_params}")
        print('======' * 10)
        print('======' * 10)

    return model

########################################
########################################

def convnext_arch(image_size, num_classes, verbose=False): 
    # Model Definition
    # if other ConvNexT models are desired change 'convnext_tiny' to 'convnext_base', 'convnext_small'
    model = timm.create_model('convnext_tiny', pretrained=True, 
                              num_classes=num_classes,
                            drop_rate=0.1        # embedding & MLP dropout
                              )
    
    # Get transforms for that size
    config = resolve_data_config({'img_size': image_size}, model=model)    
    # print("Transform input size:", config['input_size'])

    # Step 3: Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Step 4: Unfreeze the classifier parameters
    for param in model.head.parameters():
        param.requires_grad = True

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print('======' * 10)
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Frozen parameters: {total_params - trainable_params}")
        print('======' * 10)
        print('======' * 10)

    return model

########################################
########################################



########################################
################################################################################
########################################
##### ############# Training and Validation Loop ############

# Training Loop
def train_model(model, num_epochs, train_loader, val_loader, 
                device, criterion, optimizer, scheduler, 
                patience_epochs_early_stop, model_save_path):
    best_val_loss = float('inf')
    # Lists to store losses
    train_losses = []
    val_losses = []

    train_ACCs = []
    val_ACCs = []
    train_F1s, val_F1s = [], []
    train_AUCs, val_AUCs = [], []

    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        train_ACCs.append(train_acc)

        val_loss, val_acc, val_f1, val_auc = validate_model(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        val_ACCs.append(val_acc)
        val_F1s.append(val_f1)
        val_AUCs.append(val_auc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val f1-score:{val_f1:.2f}%, Val AUC:{val_auc:.2f}%" )
        
        if val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy() # Save the best model state
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Best model saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1

        # Check if scheduler will reduce LR
        old_lr = optimizer.param_groups[0]['lr']
        print("current LR:  ", scheduler.get_last_lr()[0])        
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # If LR was reduced, restore best model weights
        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr} to {new_lr}")
            print("Restoring best model weights...")
            model.load_state_dict(best_model_state)

        # Save plots every 5 epochs
        if (epoch + 1) % 5 == 0:  
            plot_train_curve(train_losses, val_losses, "Loss", model_save_path)
            plot_train_curve(train_ACCs, val_ACCs, "Accuracy", model_save_path)
            plot_train_curve(train_F1s, val_F1s, "F1", model_save_path)
            plot_train_curve(train_AUCs, val_AUCs, "AUC", model_save_path)


        if epochs_no_improve > patience_epochs_early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1} (no val-loss improvement for {patience_epochs_early_stop} epochs).")
            print(f"✅ Best model saved at epoch {best_epoch}")

            plot_train_curve(train_losses, val_losses, 'Loss', model_save_path)
            plot_train_curve(train_ACCs, val_ACCs, 'Accuracy', model_save_path)
            plot_train_curve(train_F1s, val_F1s, 'F1', model_save_path)
            plot_train_curve(train_AUCs, val_AUCs, 'AUC', model_save_path)

            save_metrics(train_losses, val_losses,
                 train_ACCs, val_ACCs,
                 train_F1s, val_F1s,
                 train_AUCs, val_AUCs,
                 model_save_path)

            break

    plot_train_curve(train_losses, val_losses, "Loss", model_save_path)
    plot_train_curve(train_ACCs, val_ACCs, "Accuracy", model_save_path)
    plot_train_curve(train_F1s, val_F1s, "F1", model_save_path)
    plot_train_curve(train_AUCs, val_AUCs, "AUC", model_save_path) 

    save_metrics(train_losses, val_losses,
        train_ACCs, val_ACCs,
        train_F1s, val_F1s,
        train_AUCs, val_AUCs,
        model_save_path)
    
    print(f"✅ Best model saved at epoch {best_epoch}")


# Validation Loop
def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            # Probabilities for AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 probabilities

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = float('nan')  # In case there's only one class in val set

    #print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    return avg_val_loss, val_acc, val_f1, val_auc



############# plots ############

# Plotting
def plot_train_curve(train_values, val_values, ylabel, model_save_path):
    plt.figure(figsize=(10, 6))
    if len(train_values)>0:
        plt.plot(train_values, label="Train "+ylabel)
    if len(val_values)>0:
        plt.plot(val_values, label="Val " + ylabel)
    if ylabel=='Loss':
    	pass
    elif ylabel=='Accuracy':
        plt.ylim(-0.1, 101)      
    else:
	    plt.ylim(-0.1, 1.1)  
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " Curve")
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(model_save_path[:-4]+'_'+ylabel+'.pdf')
    plt.close()  # Close the figure to free memory

    # Pause briefly to ensure the file is saved
    #time.sleep(1)   # 0.5 seconds is usually plenty


def save_metrics(train_losses, val_losses,
                 train_ACCs, val_ACCs,
                 train_F1s, val_F1s,
                 train_AUCs, val_AUCs,
                 model_save_path):
    
    # Save metrics to pickle
    metrics_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ACCs": train_ACCs,
        "val_ACCs": val_ACCs,
        "train_F1s": train_F1s,
        "val_F1s": val_F1s,
        "train_AUCs": train_AUCs,
        "val_AUCs": val_AUCs
    }

    metrics_save_path = model_save_path[:-4] + '_metrics.pkl'
    with open(metrics_save_path, "wb") as f:
        pickle.dump(metrics_dict, f)
    print(f"Metrics saved to {metrics_save_path}")

