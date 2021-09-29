from __future__ import print_function
import argparse
import os
import shutil
import time
import scipy.misc as misc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import utils
import SRNet 
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch of SRNet')
parser.add_argument('train_cover_dir', type=str, metavar='PATH',\
		    help='path of directory containing all' + \
			 'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',\
		    help='path of drectory containing all' + \
			 'training stego images')
parser.add_argument('valid_cover_dir', type = str, metavar='PATH',\
		    help = 'path of directory containing all' + \
			   'validation cover images')
parser.add_argument('valid_stego_dir', type=str, metavar='PATH',\
		    help='path of directory containing all' +\
			 'validation stego images')
parser.add_argument('--batch-size', type=int, default=32,metavar='N',\
		    help='input batch size for training(default:32)')
parser.add_argument('--test-batch-size', type=int, default=100,metavar='N',\
		    help='input batch size for testing(default:100)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',\
		    help='number of epochs to train (default:1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
		    help='learning rate (default: 4e-1)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,\
		    help='use batch normalization after each activation,'+
			 'also disable pair constraint (default: False)')
parser.add_argument('--embed-otf', action='store_true', default=False,\
		    help='use beta maps and embed on the fly instead'+\
			 'of use stego images (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,\
		    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,\
		    help='index of gpu used(default:0)')
parser.add_argument('--log-interval', type=int, default=175, metavar='N',\
		    help='how many batchs to wait before loging training')
parser.add_argument('--seed', type=int, default=1, metavar='s',
		    help='random seed (default:1)')


args = parser.parse_args()
arch = 'YeNet_with_bn' if args.use_batch_norm else 'YeNet'
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
else:
    args.gpu = None
kwargs = {'num_workers':8, 'pin_memory': True} if args.cuda else {}

train_transform = transforms.Compose([
	    utils.RandomRot(),
	    utils.RandomFlip(),
	    utils.ToTensor()])

valid_transform = transforms.Compose([
	    utils.ToTensor()])

print('Generate loaders.....')
train_loader=utils.DataLoaderStego(args.train_cover_dir,\
				   args.train_stego_dir, shuffle=False,\
				   embedding_otf=args.embed_otf,\
				   pair_constraint=not(args.use_batch_norm),
				   batch_size = args.batch_size,\
				   transform = train_transform,
			           num_workers = kwargs['num_workers'],
				   pin_memory = kwargs['pin_memory'])
valid_loader = utils.DataLoaderStego(args.valid_cover_dir,\
				     args.valid_stego_dir, shuffle=True,\
				     embedding_otf=False,\
				     pair_constraint=True,\
				     batch_size=args.test_batch_size,\
				     transform=valid_transform,\
				     num_workers=kwargs['num_workers'],
				     pin_memory=kwargs['pin_memory'])
print('train_loader have {} iterations, valid_loader have {} iterations'\
      .format(len(train_loader), len(valid_loader)))
      
print('Generate model')
net = SRNet.SRNet()

print('Generate loss and optimizer')
if args.cuda:
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adamax(net.parameters(), lr=args.lr)

def train(epoch):
    net.train()
    running_loss = 0.
    running_accuracy = 0.
    for batch_idx, data in enumerate(train_loader):
        images, labels = Variable(data['images']), Variable(data['labels'])
        imgsize = images.size()
        images = images.view(imgsize[0]*imgsize[1],imgsize[2],imgsize[3],\
        imgsize[4])
        labels = labels.view(imgsize[0]*imgsize[1] ,imgsize[2])
        labels = labels.squeeze_()
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        accuracy = SRNet.accuracy(outputs, labels).item()
		
        running_accuracy += accuracy
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval
            print(('Train epoch\epochs: [{}/{}] [{}/{}]  Accuracy: '+
                   '{:.2f}%  Loss: {:.6f}').format(\
                   epoch, args.epochs, batch_idx+1, len(train_loader),\
                   100 * running_accuracy, running_loss))
            running_loss = 0.
            running_accuracy = 0.


def valid():
    net.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_accuracy = 0.
        correct = 0

        for data in valid_loader:
            images, labels = Variable(data['images']), Variable(data['labels'])
            imgsize = images.size()
            images = images.view(imgsize[0]*imgsize[1],imgsize[2],\
                                 imgsize[3],imgsize[4])
            labels = labels.view(imgsize[0]*imgsize[1],imgsize[2]).squeeze_()
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            valid_loss += criterion(outputs, labels).item()
            valid_accuracy += SRNet.accuracy(outputs, labels).item()
        valid_loss /= len(valid_loader)
        valid_accuracy /= len(valid_loader)
        print('valid-set: Loss: {:.4f}, Accuracy: {:.2f}%'.format(
              valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy

def save_checkpoint(state,  filename='logs/checkpoint.pth.tar'):
    torch.save(state, filename)



_time = time.time()
for epoch in range(1, args.epochs + 1):
    print('--'*30)
    train(epoch)
    valid()
    save_checkpoint({
        'epoch': epoch,
        'arch': arch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()})
    print('Training times: {:.2f}\n'.format(time.time()-_time))
    _time = time.time()


'''
	print('Test')
	_,accuracy = valid()
	if accuracy > best_accuracy:
		best_accuracy = accuracy
		is_best = True
	else:
		is_best = False
	print('Time:', time.time() - _time)
	save_checkpoint({
		'epoch': epoch,
		'arch': arch,
		'state_dict': net.state_dict(),
		'best_prec1': accuracy,
		'optimizer': optimizer.state_dict()}, is_best)

'''
