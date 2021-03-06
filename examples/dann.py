import random
import time
import warnings
import sys
import argparse
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('.')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
from dalib.adaptation.ssl import SSL
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy, ForeverDataIterator
from tools.transforms import ResizeImage
from tools.lr_scheduler import StepwiseLR
from tools.dataloaders import get_dataloader


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]

    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    val_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=val_transform)
    val_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)

    train_source_loader = get_dataloader(train_source_dataset, start_fraction=0, stop_fraction=0.8,
                                         shuffle=True, batch_size=args.batch_size, num_workers=args.workers,
                                         drop_last=True)
    train_target_loader = get_dataloader(train_target_dataset, start_fraction=0, stop_fraction=0.8,
                                         shuffle=True, batch_size=args.batch_size, num_workers=args.workers,
                                         drop_last=True)
    val_loader = get_dataloader(val_target_dataset, start_fraction=0.8, stop_fraction=1,
                                shuffle=True, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)

    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, evaluate=True, download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, train_source_dataset.num_classes).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + domain_discri.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # define loss function
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(best_model)
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    if args.vwhl:
        print("Start VWHL")

        inferred_dataloader = get_ssl_dataloader("train", train_source_loader, train_target_loader, classifier, args)

        lr_scheduler2 = StepwiseLR(optimizer, init_lr=args.vwhl_lr, gamma=0.001, decay_rate=0.75)

        if args.vwhl_finetuning:
            print("Using VWHL finetuning")
            # we are trying out fine-tuning now
            classifier2 = ImageClassifier(backbone, inferred_dataloader.dataset.num_classes).to(device)
        else:
            classifier2 = classifier

        # evaluate on test set
        acc2 = validate(test_loader, classifier2, args)
        print("test_acc2 = {:3.1f}".format(acc2))

        # SSL
        best_acc2 = 0.
        for epoch in range(args.epochs):
            # train for one epoch
            train_ssl(inferred_dataloader, classifier2, optimizer, lr_scheduler2, epoch, args)

            # evaluate on validation set
            acc2 = validate(val_loader, classifier2, args)

            # remember best acc@2 and save checkpoint
            if acc2 > best_acc2:
                best_model2 = copy.deepcopy(classifier2.state_dict())
            best_acc2 = max(acc2, best_acc2)

        print("best_acc2 = {:3.1f}".format(best_acc2))

        # evaluate on test set
        classifier2.load_state_dict(best_model2)
        acc2 = validate(test_loader, classifier2, args)
        print("test_acc2 = {:3.1f}".format(acc2))


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_adv: DomainAdversarialLoss, optimizer: SGD,
          lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        lr_scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def get_ssl_dataloader(purpose: str, source_loader: DataLoader, target_loader: DataLoader, model: ImageClassifier,
                       args: argparse.Namespace) -> DataLoader:
    ssl = SSL(purpose, target_dataloader=target_loader, source_dataloader=source_loader,
              percentile_rank=args.ssl_percentile_rank, weight_inferred_dataset=args.ssl_weight_inferred_dataset)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(target_loader):
            images = images.to(device)

            # compute output
            output, _ = model(images)

            ssl.add_predictions(output)

    inferred_dataloader = ssl.get_semi_supervised_dataloader()

    return inferred_dataloader


def train_ssl(inferred_dataloader: DataLoader, model: ImageClassifier,
              optimizer: SGD, lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(inferred_dataloader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, labels) in enumerate(inferred_dataloader):
        lr_scheduler.step()

        x = x.to(device)
        labels = labels.to(device)

        # compute output
        output, _ = model(x)
        loss = F.cross_entropy(output, labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--vwhl', action='store_true', default=False,
                        help='Set this flag if you want to use VWHL (SSL step after training)')
    parser.add_argument('--vwhl-finetuning', action='store_true', default=False,
                        help='Set this flag if you want to use the DC-trained model during SSL in VWHL')
    parser.add_argument('--vwhl-lr', default=0.01, type=float,
                        metavar='VWHL LR', help='initial learning rate for VWHL')
    parser.add_argument('--ssl_percentile_rank', default=0, type=float,
                        help='Percentile rank required to accept labels to inferred dataset '
                             'during semi-supervised learning.')
    parser.add_argument('--ssl_weight_inferred_dataset', default=1, type=float,
                        help='Weight of the inferred dataset in target domain while '
                             'merging with the source domain dataset')

    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        from nvsmpy import CudaCluster

        cluster = CudaCluster()
        with cluster.limit_visible_devices(max_n_processes=1):
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main(args)
