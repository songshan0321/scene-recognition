import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils

parser = argparse.ArgumentParser(description='PyTorch Places3 Testing')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('model', metavar='MODEL',
                    help='path to model checkpoint')

def main():
    global args
    args = parser.parse_args()
    print(args)

    # Writer for Tensorboard
    global writer
    log_dir = "logs/test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

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
    global classes
    classes = {v: k for k, v in test_data.class_to_idx.items()}
    print(classes)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=16, shuffle=True,
        num_workers=1, pin_memory=True)

    
    # Load model
    # model_file = 'checkpoint/alexnet_best.pth.tar'
    model_file = args.model
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
            if len(input_var) < 9:
                continue
            writer.add_figure('predictions vs. actuals',
                plot_classes_preds(model, input_var, target),
                global_step=i)

    print(' Loss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}'
          .format(loss=losses, top1=top1))

    return top1.avg

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(9):
        ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx].item()]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    ## Code to test the merging of escalator and staircase 
    # pred[pred==2] = 0
    # target[target==2] = 0
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # for k in topk:
    correct_k = correct[:1].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()