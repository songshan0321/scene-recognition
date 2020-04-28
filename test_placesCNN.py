import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils

parser = argparse.ArgumentParser(description='PyTorch Places3 Testing')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

def main():
    global args
    args = parser.parse_args()
    print(args)

    # Writer for Tensorboard
    global writer
    writer = SummaryWriter('test/places3')

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data loading code
    testdir = os.path.join(args.data, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_data = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    print(test_data.class_to_idx)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)

    
    # Load model
    model_file = 'alexnet_best.pth.tar'
    model = models.__dict__['alexnet'](num_classes=3)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # Testing
    prec1 = test(test_loader, model, criterion)

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

def test(loader, model, criterion):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            target = target.cuda(async=True)
            input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # print(input_var.device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # for k in topk:
    correct_k = correct[:1].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res




if __name__ == '__main__':
    main()