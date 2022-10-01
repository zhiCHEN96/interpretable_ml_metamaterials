import argparse
import os
import numpy as np
import shutil
from scipy import io
import time
import random
import pandas
import scipy.io as sio

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

from sklearn.model_selection import train_test_split
from utils import bg_existence_labels, vector2symmatrix

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--ngpu', default=1, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='random seed (default: 1234)')
parser.add_argument("--prefix", type=str, default = 'resnet18', metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--bandgap', default=0, type=int, metavar='BG',
                    help='index of frequency range used to create bandgap labels')
best_prec1 = 0
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, prefix):
    filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)

def accuracy(output, target):
    return ((output.view(-1)>0).long() == target).float().mean() * 100.0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_weights_for_balanced_classes(images, nclasses):
    """Get the weights of samples so that they can be sampled in balance"""                      
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))
    print(count)                       
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]
                                
    return weight

def get_loader_from_dataset(dataset):
    """Get balanced sampler"""
    weights = make_weights_for_balanced_classes(dataset, 2)                                                                
    weights = torch.DoubleTensor(weights)                                   
    sampler = Data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                    
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle = sampler==None,                              
        sampler = sampler, num_workers=args.workers, pin_memory=True)
    return loader

def get_loaders():
    """Get the data loaders"""
    # load the data
    data = sio.loadmat('./data/bandgap_data.mat')
    x = data['feature_raw'] # raw features
    dispersion = data['dispersion'] # dispersion curves

    freq_ranges = np.array([[0,1000],[1000,2000],[2000,3000],[3000,4000],[4000,5000]]) # target ranges
    labels = bg_existence_labels(dispersion, freq_ranges)

    y = labels[:,args.bandgap].astype('int32')
    print("freq range: ", freq_ranges[args.bandgap])
    x_mat = vector2symmatrix(x)
    n = x_mat.shape[0]
    idx = list(range(n))
    x_train, x_test, y_train, y_test = train_test_split(x_mat, y, test_size=0.2)
    dataset_train = Data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).long())
    dataset_test = Data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())
    
    loader_train = get_loader_from_dataset(dataset_train)
    loader_test = get_loader_from_dataset(dataset_test)
    loader_plot = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle = True,                              
        sampler = None, num_workers=args.workers, pin_memory=True)
    return loader_train, loader_test, loader_plot

def train(loader_train, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(loader_train):
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).unsqueeze(1).cuda()
        target_var = torch.autograd.Variable(target).unsqueeze(1).float().cuda()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(loader_train), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

def validate(loader_val, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(loader_val):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).unsqueeze(1).cuda()
        target_var = torch.autograd.Variable(target, volatile=True).unsqueeze(1).float().cuda()
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(loader_val), batch_time=batch_time, loss=losses,
                   top1=top1))
    
    print(' * Prec@1 {top1.avg:.3f}'
            .format(top1=top1))

    return top1.avg

def main():
    global args, best_prec1
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    model = torchvision.models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
    model.fc = nn.Linear(512 * 1, 1)
    model.cuda()
    
    loader_train, loader_val, loader_test = get_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCEWithLogitsLoss()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    else:
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch)
            
            # train for one epoch
            train(loader_train, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            prec1 = validate(loader_val, model, criterion, epoch)
            
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            print(best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.prefix)


if __name__ == '__main__':
    main()

