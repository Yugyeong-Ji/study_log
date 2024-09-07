import os
import torch
from tqdm import tqdm
from .validate_engine import validate_engine
from metrics.utils import  save_labels_and_predictions

def train_engine(args, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler):
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for features, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        val_loss, val_labels,val_predictions = validate_engine(args, model, val_dataloader, criterion)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {(train_loss/len(train_dataloader)):4f}, Validation Loss: {val_loss:4f}, Learning Rate: {current_lr:.6f}')
        
        # Update the learning rate based on the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)  # ReduceLROnPlateau requires the validation loss
        else:
            scheduler.step()  # Other schedulers just step after each epoch

        # Save the best model
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))
            save_labels_and_predictions(args, val_labels, val_predictions)
        # Save the last model
        torch.save(model.state_dict(), os.path.join(args.save_path, 'last_model.pt'))
