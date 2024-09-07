import os
import torch
from tqdm import tqdm
from metrics import load_metric
from engine import validate_engine
from metrics.utils import print_score, save_labels_and_predictions

def train_engine(args, model, train_dataloader, val_dataloader, criterion, optimizer,scheduler):
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        all_labels = None
        all_predictions = None
        
        for features, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            outputs = model(features)
            try:
                ouputs = outputs.squeeze()
            except:
                pass
            # print(f"outputs & labels shape : {outputs.shape, labels.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_labels = torch.cat([all_labels, labels.cpu()]) if all_labels is not None else labels.cpu()
            all_predictions = torch.cat([all_predictions, outputs.cpu()]) if all_predictions is not None else outputs.cpu()

        evaluate = load_metric(args)
        train_score = evaluate(args, all_labels,all_predictions)
        val_loss, val_score,val_labels,val_predictions = validate_engine(args, model, val_dataloader, criterion)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {(train_loss/len(train_dataloader)):4f}, Validation Loss: {val_loss:4f}')
        
        
        print_score("train score", train_score)
        print_score("val score", val_score)
        
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
