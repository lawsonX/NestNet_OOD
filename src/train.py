import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import torch.nn.functional as F


from models import ResNet20, ResNet18,ResNetEns
from class_balanced_loss import CB_loss
from util import *
from utils import Bar, AverageMeter, accuracy

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")
# Datasets
parser.add_argument("-m", "--model", default="resnet18", type=str, choices=["resnet18, resnet20"])
parser.add_argument("-d", "--dataset", default="cifar10", type=str)
parser.add_argument("-j","--workers",default=4,type=int,metavar="N",help="number of data loading workers (default: 4)",)
# Optimization options
parser.add_argument("--global_rounds",default=40,type=int,metavar="N",help="number of total epochs to run")
parser.add_argument("--local_epochs",default=4,type=int,metavar="N",help="number of total epochs to run")

parser.add_argument("--train-batch", default=128, type=int, metavar="N", help="train batchsize")
parser.add_argument("--test-batch", default=100, type=int, metavar="N", help="test batchsize")
parser.add_argument("--lr","--learning-rate",default=0.01,type=float,metavar="LR",help="initial learning rate",)
parser.add_argument('--schedule', type=int, nargs='+', default=[45, 80],help='Decrease learning rate at these epochs.')
parser.add_argument("--weight-decay","--wd",default=5e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",)

parser.add_argument("--num_branches", type=int, default=2, help="number of experts model")
parser.add_argument("--beta", type=float, default=0.9999)
parser.add_argument("--gama", type=float, default=1.0)

parser.add_argument("--prune-mode", type=str, default='global', help="global(per weight) or local (layer-wise per channel) ", choices=['global', 'local'])
parser.add_argument("--prune-rate", type=float, default=0.05)
parser.add_argument("--base", type=float, default=0.5, help='the base accuracy to start pruning')
parser.add_argument("--step", type=float, default=0.02, help='increase base by step every time the model is pruned')
parser.add_argument("--pretrained", dest='pretrained', action='store_true')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("-c","--checkpoint",default="checkpoint/test/",type=str,metavar="PATH",help="path to save checkpoint (default: checkpoint)",)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data
    print("==> Preparing dataset %s" % args.dataset)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    print("data loaded\n")
    
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        OODloader = datasets.CIFAR100
        num_classes = 6
    else:
        dataloader = datasets.CIFAR100
        OODloader = datasets.CIFAR10
        num_classes = 100
        
    global num_sub
    num_sub = num_classes//args.num_branches
    
    global num_class
    num_class = num_classes

    trainset = dataloader(
        root="/home/xiaolirui/workspace/data", train=True, download=True, transform=transform_train
    )
    trainloader = data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=4
    )

    testset = dataloader(
        root="/home/xiaolirui/workspace/data", train=False, download=False, transform=transform_test
    )
    testloader = data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=4
    )

    OODtestset = OODloader(root='/home/xiaolirui/workspace/data', train=False, download=True, transform=transform_test)
    OODtestloader = data.DataLoader(OODtestset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("building model...\n")
    if args.model == "resnet20":
        global_model = ResNet20(num_classes=num_classes)
    elif args.model == "resnet18":
        global_model = ResNet18(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model selected")
    
    if args.pretrained:
        ckpt = torch.load('checkpoint/layer_prune/0426/res18_noprune/model_79.pth')
        global_model.load_state_dict(ckpt)
        print('pretrained global model loaded')

    global_model.to(device)
    global_model.train()

    if args.evaluate:
        criterion = nn.CrossEntropyLoss()
        models = [global_model for _ in range(args.num_branches)]
        ensemble_model = ResNetEns(models)
        ckpt = torch.load('checkpoint/test/EnsembleResnet18.pth')
        ensemble_model.load_state_dict(ckpt)
        print('\nEvaluation only')
        probID = test_prob(testloader, ensemble_model, criterion, 0, device)
        probOOD = test_prob(OODtestloader, ensemble_model, criterion, 0, device)
        print(f'"ID":{probID},"OOD":{probOOD}')
        # sio.savemat(args.resume,{"ID":probID,"OOD":probOOD})

        #test_sub(testloader, model.module, criterion, start_epoch, use_cuda)
        #test_OOD(testloader, model.module, criterion, start_epoch, use_cuda)
        #test_OOD(OODtestloader, model.module, criterion, start_epoch, use_cuda)
        
        #test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        #print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return
    
    
    lr = args.lr
    # make masks
    masks = [make_mask(global_model) for _ in range(args.num_branches)]
    mats_score = [make_grad_mat(global_model) for _ in range(args.num_branches)]
    args.base_list = [args.base for _ in range(args.num_branches)]
    for epoch in range(args.global_rounds):
        if epoch in args.schedule:
            lr = lr / 10
        print("-------------Training Global Epoch:{}/{} with learning rate:{}-------------".format(epoch + 1, args.global_rounds, lr))
        models, global_model, masks, mats_score = train_n_val(
            args,global_model,masks,mats_score,
            trainloader,testloader,
            num_branches=args.num_branches,
            num_epochs=args.local_epochs,
            device=device,
            lr=lr,
            )
        ensemble_model = ResNetEns(models)
        rest_losses, test_top1 = test(testloader,ensemble_model,epoch,device)
        test_top1 = test_top1/100
        
        if args.prune_mode == 'global':
            remain_ratio = 0
            if test_top1 > args.base:
                # masks = global_pruning_v2(masks, mats_score, models, additional_pruning_rate=0.1)
                masks = global_pruning_v1(masks, mats_score, pruning_ratio = args.prune_rate)
                args.base += args.step
                print(f'Mask pruned globally by 10% | Increase prune base accuracy to:{round(float(args.base),2)}')
            
            for i in range(len(masks)):
                remain_ratio += compute_sparsity(masks[i])
            remain_ratio = remain_ratio / len(masks)
            print(f"\nGlobaln Validation Loss avg :{round(rest_losses, 4)} | Acc avg :{round(test_top1,4)} | Current Pruned ratio:{round((1-remain_ratio.item()),3)} \n")
                
        # save global model and branch_masks
        save_checkpoint(
            global_model.state_dict(),
            checkpoint=args.checkpoint,
            filename=f"global_{epoch}.pth",
        )
        save_checkpoint(masks, checkpoint=args.checkpoint, filename=f"mask_{epoch}.pth")
    
    # saved the ensembel model when finshed training
    save_checkpoint(
            ensemble_model.state_dict(),
            checkpoint=args.checkpoint,
            filename=f"EnsembleResnet18.pth",
        )

def train_n_val(args, global_model, masks, mats_score, trainloader, testloader, num_branches, num_epochs, device, lr):
    criterion = nn.CrossEntropyLoss()
    models = []
    classifier_weight_list = [[] for _ in range(num_branches)]
    val_loss, val_acc = [], []
    for idx in range(num_branches):
        branch_model = create_masked_branch_model(
            global_model, classifier_weight_list[idx], masks[idx]
        )
        branch_optimizer = optim.SGD(branch_model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                out = branch_model(inputs)
                temp_target = targets.clone() 
                # for t in range(len(temp_target)):
                #     if temp_target[t] >= idx*num_sub and temp_target[t] <= idx*num_sub+num_sub-1:
                #         temp_target[t] = temp_target[t]-idx*num_sub
                #     else:
                #         temp_target[t] = num_sub
                # temp_target_onehot = F.one_hot(temp_target, num_sub+1).float()
                # for t in range(len(temp_target)):
                #     if temp_target[t]== num_sub:
                #         temp_target_onehot[t,:-1] = 1/num_class
                #         temp_target_onehot[t,num_sub] = (num_class-num_sub)/num_class

                if args.num_branches == 2:
                    for temp in range(len(temp_target)):
                        if temp_target[temp]>=idx*5 and temp_target[temp]<=idx*5+4:
                            temp_target[temp] = temp_target[temp]-idx*5
                        else:
                            temp_target[temp] = 5
                elif args.num_branches == 10:
                    for temp in range(len(temp_target)):
                        if temp_target[temp] == idx:
                            temp_target[temp] = 0
                        else:
                            temp_target[temp] = 1
                else:
                    pass

                # loss_cb = CB_loss(
                #     temp_target_onehot,
                #     out,
                #     [5000, 45000],
                #     2,
                #     "sigmoid",
                #     0.9999,
                #     1.0,
                # )
                loss_cb = CB_loss(temp_target,out,[5000,5000,5000,5000,5000,250000],6,"focal",args.beta,args.gama)
                loss_ce = criterion(out, temp_target)
                loss = loss_ce + loss_cb
                
                branch_optimizer.zero_grad()
                loss.backward()
                branch_optimizer.step()
                _, preds = torch.max(out, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == temp_target)
            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset)
            print("Training Branch %d | Local Epoch [%d/%d] | Train loss: %.4f | Train Acc: %.4f"% (idx, epoch + 1, num_epochs, epoch_loss, epoch_acc))
        
        # 计算score
        for name, param in branch_model.named_parameters():
            if "conv" in name and "weight" in name:
                mats_score[idx][name] = param.grad.data # for # global_pruning_v2 接收grad
                mats_score[idx][name] = param.data * param.grad.data  # global_pruning_v1 输入计算好的score, 使用v1是需要注释本行和下行
                mats_score[idx][name] = torch.sum(mats_score[idx][name], dim=tuple(range(1, len(mats_score[idx][name].shape))))

        models.append(branch_model)
        classifier_weight_list[idx] = branch_model.fc.state_dict()
        
        # validate branch
        avg_val_loss, avg_val_acc = validate_branch(branch_model, testloader, idx, device, grad=False)
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)
        print("Validating Branch %d | Val loss: %.4f | Val Acc: %.4f"% (idx, avg_val_loss, avg_val_acc))
        
        # 本地layer-wise剪枝
        if args.prune_mode == 'local':
            # prune branch mask
            if avg_val_acc > args.base_list[idx] and round((1-remain_ratio.item()),3) < 0.3 :
                for name, param in branch_model.named_parameters():
                    if "conv" in name and "weight" in name:
                        prune_mask_layerwise(param.grad.data, param.data, masks[idx][name], args.pruning_rate)
                args.base_list[idx] += args.step
                print(f'Branch:{idx} is pruned by {args.pruning_rate} | Increase prune base to:{round(float(args.base_list[idx]),2)}')
            
            remain_ratio = compute_sparsity(masks[idx])
            print(f'Current Pruned ratio:{round((1-remain_ratio.item()),3)}')

    # Update weight for global model
    global_model = update_global_model_v2(global_model, models, masks, alpha=0.1)
    # global_model = update_global_model_v1(global_model, models, masks, device, alpha=0.1)

    return models, global_model, masks, mats_score


def validate_branch(branch_model, testloader, idx, device, grad=False):
    branch_model.eval()
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = branch_model(inputs)
        temp_target = targets.clone()
        for temp in range(len(temp_target)):
            if temp_target[temp] >= idx * 5 and temp_target[temp] < (idx + 1) * 5:
                temp_target[temp] = temp_target[temp] % 5
            else:
                temp_target[temp] = 5
        loss = criterion(out, temp_target)
        if grad:
            loss.backward()
        _, preds = torch.max(out, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == temp_target)

    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_corrects.double() / len(testloader.dataset)

    return epoch_loss, epoch_acc

def test(testloader, model, epoch, device):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs,_ = outputs
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    return (losses.avg, top1.avg)

def test_prob(testloader, model, criterion, epoch, device):

    model.eval()
    model = model.to(device)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
        outputs,output_list = model(inputs)

        max_pred = torch.cat(tuple(out.unsqueeze(2) for out in output_list),2)
        #max_pred,_ = torch.max(pred,dim=1)

        if batch_idx==0:
            max_pred_combine = max_pred.detach().cpu().numpy()
        else:
            max_pred_combine = np.concatenate((max_pred_combine,max_pred.detach().cpu().numpy()))

        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | ETA: {eta:}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    )
        bar.next()
    bar.finish() 

    return max_pred_combine

if __name__ == "__main__":
    main()
