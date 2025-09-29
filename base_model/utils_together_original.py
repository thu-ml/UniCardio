import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import random
import csv
class SimpleCSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # If the file doesn't exist, create one with headers.
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['epoch', 'metric1', 'value1', 'metric2', 'value2', 'stage', 'threshold', 'lr'])

    def log(self, epoch, metric1, value1, metric2, value2, stage, threshold, lr):
        with open(self.filepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, metric1, value1, metric2, value2, stage, threshold, lr])
class thresholder:
    @staticmethod
    def forward(batch_number):
        if  batch_number <= 200:
            stage = 1 
            threshold = 0
        elif batch_number > 200 and batch_number <= 250:
            stage = 2
            threshold = 0.5
        elif batch_number > 250 and batch_number <= 350:
            stage = 2
            threshold = 0.5
        elif batch_number > 350 and batch_number <= 400:
            stage = 2
            threshold = 0.5
        elif batch_number > 400 and batch_number <=450:
            stage = 3 
            threshold = 0.5
        elif batch_number > 450 and batch_number <=550:
            stage = 3 
            threshold = 0.5
        elif batch_number > 550 and batch_number <=600:
            stage = 3 
            threshold = 0.5
        elif batch_number > 600 and batch_number <=800:
            stage = 3 
            threshold = 2/3
        return stage, threshold
            
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername="",
    device = 'cuda',
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    foldername = "./check"
    if foldername != "":
        os.makedirs(foldername, exist_ok=True)
        log_path = os.path.join(foldername, 'loss.csv')
        logger = SimpleCSVLogger(log_path)

    p1 = int(0.18 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    for epoch_no in range(config["epochs"]):
        stage, train_threshold = thresholder.forward(epoch_no)
        avg_loss1 = 0
        avg_loss2 = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                task_dice = np.random.rand(1)
                dirty_dice = np.random.rand(1)
                condition_dice = np.random.rand(1)
                # loss1 = model(train_batch[0].to(device), train_batch[1].to(device), train_batch[2].to(device), train_batch[3].to(device), task_dice,  dirty_dice, condition_dice, train_threshold, stage, is_train=1, train_gen_flag = 0)
                loss1 = model(
                    train_batch[0].to(device), 
                    sig_impute=train_batch[1].to(device),
                    sig_denoise=train_batch[2].to(device),
                    mask=train_batch[3].to(device),
                    task_dice=task_dice,
                    dirty_dice=dirty_dice,
                    condition_dice=condition_dice,
                    train_threshold=train_threshold,
                    stage=stage,
                    is_train=1,
                    train_gen_flag=0
                    )
                loss1 = loss1.mean(dim = 0)
                loss1.backward()
                avg_loss1 += loss1.item()/train_batch[0].shape[0]
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss1": avg_loss1 / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()
        my_lr = lr_scheduler.get_lr()
        logger.log(epoch_no, 'loss_noise', avg_loss1, 'loss_rmse', avg_loss2, stage, train_threshold, my_lr)
        if valid_loader is not None and (epoch_no + 1) % 1 == 0:
            if foldername != "":
                output_path = os.path.join(foldername, f'model{epoch_no}.pth')
                torch.save(model.state_dict(), output_path)
    
    
    if foldername != "":
        output_path = foldername + "/final.pth"
        torch.save(model.state_dict(), output_path)