import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import shutil 
from models_normal_attention import *
from dataset_normal_attention import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 

from opts import parse_opts_offline
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import pdb
import time
import datetime
import pathlib

print("CnnLSTM+Attention")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def save_checkpoint_mid(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_30_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_30_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

best_prec1 = 0

if __name__ == '__main__':
    opt = parse_opts_offline()
    
    device = torch.device(opt.torch_device)
    if opt.root_path != '':
        if opt.result_path:
            opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.annotation_path:
            opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
        if opt.video_path:
            opt.video_path = os.path.join(opt.root_path, opt.video_path)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    if not opt.no_train:
        training_data = Gesturedata("train.txt")                        
        train_dataloader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])
        
    if not opt.no_val:
        validation_data = Gesturedata("valid.txt")
        test_dataloader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
        
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), 
            ['epoch','num_epochs','batch_i','loss','loss(mean)','acc'])

    cls_criterion = nn.CrossEntropyLoss().to(device)
    
    model = ConvLSTM(
        num_classes=27,
        latent_dim=128,
        lstm_layers=1,
        hidden_dim=512,
        bidirectional=True,
        attention=True,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    checkpoint_interval=1
    num_epochs=100
    
    print("completed setting")
    
    if opt.resume_path:
        print("Re-training....")
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path, map_location="cuda:0")
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        print(opt.begin_epoch, ": ", opt.no_train)
        print(opt.arch)
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        
    def test_model(epoch, num_epochs, criterion):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                model.lstm.reset_hidden_state()
                predictions = model(image_sequences)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = criterion(predictions, labels).item()
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
            val_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss,
                    'loss(mean)':np.mean(test_metrics["loss"]),
                    'acc': np.mean(test_metrics["acc"])
            })
        return np.mean(test_metrics["acc"])  
    
    def train_model(train_dataloader, criterion, metrics):
        model.train()
        prev_time = time.time()
        for batch_i, (X, y) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()
            model.lstm.reset_hidden_state()
            predictions = model(image_sequences)

            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            metrics["loss"].append(loss.item())
            metrics["acc"].append(acc)

            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(metrics["loss"]),
                    acc,
                    np.mean(metrics["acc"]),
                    time_left,
                )
            )
            train_batch_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss.item(),
                    'loss(mean)':np.mean(metrics["loss"]),
                    'acc': np.mean(metrics["acc"])
                    })
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_logger.log({
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'batch_i': batch_i,
                    'loss':loss.item(),
                    'loss(mean)':np.mean(epoch_metrics["loss"]),
                    'acc': np.mean(epoch_metrics["acc"])
                    })  
        return loss
    
    for epoch in range(num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        print("--- Epoch {epoch} ---")

        loss = train_model(train_dataloader, cls_criterion, epoch_metrics)
        
        prec1 = test_model(epoch, num_epochs, cls_criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
        save_checkpoint(state, is_best)
        
        if epoch==29:
            save_checkpoint_mid(state, is_best)
        

