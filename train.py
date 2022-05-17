import pandas as pd
import random
import torch
from torch.optim import AdamW
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

import wandb
wandb.init()

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train_loop_fn(train_dataloader, model, device, optimizer, scheduler, train_losses):
    model.train()
    tqdm_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        
        optimizer.zero_grad() # set the gradients to zero before backpropagation
        
        outputs = model(input_ids=features, 
                        attention_mask=attention_mask, 
                        labels=targets,
                        return_dict=True)
        
        loss = outputs['loss']
        loss.backward() # compute the graadients
        optimizer.step() # update the weights
        if scheduler is not None:
            scheduler.step() # change learning rate if error has almost no improvemnt
        tqdm_bar.desc = "Training loss: {:.2e}; lr: {:.2e}".format(loss.item(), scheduler.get_last_lr()[0])            
        
        if (_ % 9 == 0):
            wandb.log({'Train_batch_loss':loss.item()})
            train_losses.append(loss.item())
        
            
def eval_loop_fn(val_dataloader, model, device, val_losses):
    model.eval()
    final_targets = []
    final_logits = []
    tqdm_bar = tqdm(val_dataloader, desc="Validating", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_ids=features, 
                            attention_mask=attention_mask, 
                            labels=targets,
                            return_dict=True)
                            
        eval_loss = outputs['loss']
        val_losses.append(eval_loss.item())
        wandb.log({'Val_batch_loss':eval_loss.item()})
        # Move logits and labels to CPU
        final_targets.append(targets.detach().cpu().numpy())
        final_logits.append(outputs['logits'].detach().cpu().numpy())
        
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(final_logits, axis=0)
    flat_true_labels = np.concatenate(final_targets, axis=0)
    '''
    flat_predictions = array([
       [ 1.6957991 , -1.9192817 ],
       [ 1.8676672 , -2.2157893 ],
       [ 1.7286735 , -1.9753283 ],
       ...]], dtype=float32)
    '''

    predicted_labels = np.argmax(flat_predictions, axis=1).flatten() # perform argmax for each sample to output labels, not scores
    '''
    predicted_labels = array([
        0, 
        0, 
        0,
        ...]])
    '''
    return predicted_labels, flat_true_labels

def infer_loop_fn(unlabeled_dataloader, model, device):
    # The difference between eval_loop_fn is that we don't have true_labels now
    model.eval()
    final_logits = []
    tqdm_bar = tqdm(unlabeled_dataloader, desc="Inference", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]
        
        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_ids=features, 
                            attention_mask=attention_mask, 
                            return_dict=True)
                            
        final_logits.append(outputs['logits'].detach().cpu().numpy())
        
    # Combine the results across all batches. 
    flat_predictions = np.concatenate(final_logits, axis=0)
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten() # perform argmax for each sample to output labels, not scores
    
    return predicted_labels
    

def save_checkpoint(state, filename="saved_weights.pth"):
    # Holds the torch.Tensor objects of all the layers of the model
    # without saving the whole model architecture
    print(f"=> Saving checkpoint at epoch {state['epoch']}")
    torch.save(state,filename)


def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

def run(model, cfg, train_dataloader, val_dataloader, unlabeled_dataloader, train_losses, val_losses, accuracy_scores, F1_scores, from_checkpoint=False):
    
    device = torch.device("cuda")

    # Copy the initial model weights to the GPU.
    desc = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        eps=float(cfg['training']['adam_epsilon']), # args.adam_epsilon  - default is 1e-8.
    )

    num_epochs = cfg['training']['num_epochs'] # 1
    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # default parameter
        num_training_steps = total_steps,
    )

    if from_checkpoint:
        best_checkpoint = torch.load('/content/best_f1_state.pth')
        model.load_state_dict(best_checkpoint["state_dict"])
        last_max_F1 = best_checkpoint['F1_score']
    else:
        last_max_F1 = 0
    
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch = {epoch}")
        train_loop_fn(train_dataloader, model, device, optimizer, scheduler, train_losses)
        predicted_labels, targets = eval_loop_fn(val_dataloader, model, device, val_losses)
        F1 = f1_score(targets, predicted_labels, average='weighted') 
        acc = accuracy_score(targets, predicted_labels)
        print(f"Validation acc = {acc}; F1 = {F1}")
        F1_scores.append(F1)
        accuracy_scores.append(acc)
        wandb.log({'Validation_accuracy':acc, 'F1_score':F1})
        
        if (F1 > last_max_F1):
            print(classification_report(targets, predicted_labels))
            #print(confusion_matrix(targets, predicted_labels))
            print(sns.heatmap(confusion_matrix(targets, predicted_labels), annot=True, cmap='Blues', fmt='3g'))
            plt.show()
            last_max_F1 = F1
            checkpoint = {
                #deepcopy makes the mutable OrderedDict instance not to mutate best_state as the training goes on
                'state_dict' : deepcopy(model.state_dict()), 
                'optimizer' : deepcopy(optimizer.state_dict()),
                'epoch' : epoch,
                'accuracy' : acc,
                'F1_score' : F1,
            }
            save_checkpoint(checkpoint, filename="best_f1_state.pth")
            predicted_labels = infer_loop_fn(unlabeled_dataloader, model, device)
            best_predictions_df = pd.DataFrame(predicted_labels)
            best_predictions_df.to_csv('best_predictions_df', sep='\t', encoding='utf-8')
            best_predictions_df.to_excel("best_predictions_df.xlsx") # на всякий случай еще как excel сохраним
