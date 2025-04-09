import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from utils.utils import ce_loss
from utils.utils import *

def TrustWI_train(model,
        train_loader,
        optimizer,
        scheduler,
        device,
        model_path,
        num_classes,
        epochs,
        annealing_epoch,
        logger=None,
        model_name=None,
        evaluation=False,
        patience=25, # early stopping patience
        val_dataloader=None):
    logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    logger.info("-" * 70)
    trigger_times = 0
    last_loss = np.inf
    best_model = None
    s_t = time.time()
    for epoch_i in range(epochs):
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        for step, batch in enumerate(train_loader):
            batch_counts += 1
            optimizer.zero_grad()
            batch = batch.to(device)
            x_prop, x_prop_aug, target = batch.x_prop, batch.x_prop_aug, batch.y

            alpha_1, alpha_2, alpha_3, alpha_a = model(x_prop, x_prop_aug)
            loss = ce_loss(target, alpha_1, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_2, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_3, num_classes, epoch_i, annealing_epoch) + \
                ce_loss(target, alpha_a, num_classes, epoch_i, annealing_epoch) 
     

            loss = torch.mean(loss)
            loss.backward()

            batch_loss += loss.item()
            total_loss += loss.item()

            optimizer.step()

            if (step % 500 == 0 and step != 0) or (step == len(train_loader) - 1):
                time_elapsed = time.time() - t0_batch

                logger.info(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | "
                    f"{time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_loader)

        logger.info("-" * 70)

        min_delta = 1e-5
        if evaluation:
            val_loss, val_accuracy = TrustWI_evaluate(model, val_dataloader, num_classes, device)

            time_elapsed = time.time() - t0_epoch
            logger.info(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} |"
                f" {time_elapsed:^9.2f}")
            logger.info(f"Epoch {epoch_i+1}, Current LR: {optimizer.param_groups[0]['lr']}")
            logger.info("-" * 70)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss > last_loss - min_delta:
                trigger_times += 1
            else:
                last_loss = val_loss
                best_model = model.state_dict()
                trigger_times = 0

            if trigger_times >= patience:
                torch.save(best_model, os.path.join(model_path, f'{model_name}.h5'))
                logger.info("Early stopping")
                break

        else:
            # Early stopping
            if avg_train_loss > last_loss - min_delta:
                trigger_times += 1
            else:
                last_loss = avg_train_loss
                best_model = model.state_dict()
                trigger_times = 0

            if trigger_times >= patience:
                torch.save(best_model, os.path.join(model_path, f'{model_name}.h5'))
                logger.info("Early stopping")
                break


    e_t = time.time()
    logger.info(f'Training time: {(e_t-s_t)/60:.2f} min')
    torch.save(model.state_dict() if best_model is None else best_model, os.path.join(model_path, f'{model_name}.h5'))
    logger.info("\nTraining complete!")

def TrustWI_evaluate(model,
            val_dataloader,
            num_classes,
            device):

    model.eval()
    val_loss, val_accuracy = [], []

    for batch in val_dataloader:
        batch = batch.to(device)
        target = batch.y
        with torch.no_grad():

            alpha_1, alpha_2, alpha_3, alpha_a = model(batch.x_prop, batch.x_prop_aug)
            loss = ce_loss(target, alpha_1, num_classes, 1, 1) + \
                ce_loss(target, alpha_2, num_classes, 1, 1) + \
                ce_loss(target, alpha_3, num_classes, 1, 1) + \
                ce_loss(target, alpha_a, num_classes, 1, 1) 

            loss = torch.mean(loss)
            val_loss.append(loss.item())

            preds = torch.argmax(F.softmax(alpha_1 - 1, dim=1), dim=1)
            val_accuracy.append((preds == batch.y).cpu().numpy().mean() * 100)


    return np.mean(val_loss), np.mean(val_accuracy)

def TrustWI_predict(model,
            pse:bool,
            test_dataloader,
            device):
    model.eval()
    if pse:
        outputs, y_preds, y_true = [],[],[]
    else:
        outputs, y_preds, y_true = [],[],[]

    for batch in test_dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            alphas = model(batch.x_prop, batch.x_prop_aug)

            evidence = [alpha - 1 for alpha in alphas]

            y_true.append(batch.y)

            probs = F.softmax(evidence[3].data, dim=1) 
            outputs.append(probs)
            y_preds.append(torch.argmax(probs, dim=1))


    # Convert outputs to tensors
    y_true = torch.cat(y_true, dim=0)
    y_prob = torch.cat(outputs, dim=0)
    y_pred = torch.cat(y_preds, dim=0)


    return y_true.cpu().numpy(), y_pred.cpu().numpy(), y_prob.cpu().numpy()

def compute_dempster_shafer_uncertainty(evidence, data=None):
    if data=='data':
        evidence = F.softplus(evidence)
    alpha = evidence + 1
    belief = (alpha - 1) / torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = alpha.shape[1] / torch.sum(alpha, dim=1)

    return uncertainty.squeeze()

