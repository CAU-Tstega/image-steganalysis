from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import lr_scheduler
import utils
import XuNet

parser = argparse.ArgumentParser(description='PyTorch implementation of XuNet')
parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all' +
                         'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                    help='path of drectory containing all' +
                         'training stego images')
parser.add_argument('valid_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all' +
                         'validation cover images')
parser.add_argument('valid_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all' +
                         'validation stego images')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training(default:32)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing(default:100)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default:1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--gpu', type=int, default=0,
                    help='index of gpu used(default:0)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batchs to wait before loging training')
parser.add_argument('--seed', type=int, default=1, metavar='s',
                    help='random seed (default:1)')
parser.add_argument('--log-path', type=str, default='logs/',
                    metavar='PATH', help='path to generated log file')
args = parser.parse_args()
torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(args.gpu)
kwargs = {'num_workers': 8, 'pin_memory': True}

train_transform = transforms.Compose([
                        utils.RandomRot(),
                        utils.RandomFlip(),
                        utils.ToTensor()])

valid_transform = transforms.Compose([
                        utils.ToTensor()])
                        
print('Generate loaders.....')
train_loader = utils.DataLoaderStego(args.train_cover_dir,
                                     args.train_stego_dir, shuffle=False,
                                     batch_size=args.batch_size,
                                     transform=train_transform,
                                     num_workers=kwargs['num_workers'],
                                     pin_memory=kwargs['pin_memory'])
                                     
valid_loader = utils.DataLoaderStego(args.valid_cover_dir,
                                     args.valid_stego_dir, shuffle=True,
                                     batch_size=args.test_batch_size,
                                     transform=valid_transform,
                                     num_workers=kwargs['num_workers'],
                                     pin_memory=kwargs['pin_memory'])


def save_checkpoint(state, log_path):
    filename = log_path + 'Model_' + str(state['epoch']) + '.pth'
    torch.save(state, filename)


def train_model(model, criterion, optimizer, scheduler, log_path, num_epochs):
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 40)
        time1 = time.time()
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_acc = 0.0
            if phase == 'train':
                scheduler.step()
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            for batch_idx, data in enumerate(dataloader):
                inputs, labels = Variable(data['images']), Variable(
                    data['labels'])
                in_size = inputs.size()
                inputs = inputs.view(in_size[0] * in_size[1], in_size[2],
                                     in_size[3], in_size[4])
                labels = labels.view(in_size[0] * in_size[1], in_size[2])
                labels = labels.squeeze_()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    #print(type(inputs))
                    outputs = model(inputs)
                    accuracy = XuNet.accuracy(outputs, labels)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
                running_acc += accuracy.item()
                if phase == 'train' and (
                        batch_idx + 1) % args.log_interval == 0:
                    f1 = open(log_path + '/train_loss.txt', 'a')
                    f2 = open(log_path + '/train_acc.txt', 'a')
                    running_loss /= args.log_interval
                    running_acc /= args.log_interval
                    f1.write(str(running_loss) + '\n')
                    f2.write(str(running_acc) + '\n')
                    f1.close()
                    f2.close()
                    print('train epoch: {} [{}/{}] Loss: {:.4f} Acc: {:.4f}' \
                          .format(epoch, batch_idx + 1,
                                  len(dataloader), running_loss, running_acc))
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': arch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, log_path)

            if phase == 'val':
                time2 = time.time()
                f3 = open(log_path + '/valid_acc.txt', 'a')
                valid_loss = running_loss / len(dataloader)
                valid_acc = running_acc / len(dataloader)
                f3.write(str(valid_acc) + '\n')
                f3.close()
                print(
                    'validation: Loss: {:.4f}, Accuracy: {:.4f}, Time: {:.4f}' \
                    .format(valid_loss, valid_acc, time2 - time1))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('train_loader have {} iterations, valid_loader have {} iterations' \
          .format(len(train_loader), len(valid_loader)))

    print('Generate model')
    net = XuNet.XuNet()
    net.to(device)
    print(net)
    print('Generate loss and optimizer')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5000,
                                           gamma=0.1)

    train_model(net, criterion, optimizer, exp_lr_scheduler,
                args.log_path, args.epochs)




