#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import ResNet20, ResNet18
from class_balanced_loss import CB_loss
from utils import *

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")
# Datasets
parser.add_argument("-m", "--model", default="resnet18", type=str)
parser.add_argument("-d", "--dataset", default="cifar10", type=str)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
# Optimization options
parser.add_argument(
    "--global_rounds",
    default=60,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--local_epochs",
    default=2,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--train-batch", default=128, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--test-batch", default=100, type=int, metavar="N", help="test batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints
parser.add_argument(
    "-c",
    "--checkpoint",
    default="checkpoint/layer_prune/0426",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoint)",
)

# Architecture
parser.add_argument(
    "--num_branches", type=int, default=2, help="number of experts model"
)
parser.add_argument("--beta", type=float, default=0.9999)
parser.add_argument("--gama", type=float, default=1.0)
parser.add_argument("--pretrained", type=str, default="model_best.pth.tar")

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
    # load dataset and user groups
    # train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_cifar10_extr_noniid_v2(args.num_users, args.nclass)
    print("data loaded\n")

    dataloader = datasets.CIFAR10
    trainset = dataloader(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=4
    )

    testset = dataloader(
        root="./data", train=False, download=False, transform=transform_test
    )
    testloader = data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=4
    )

    print("building model...\n")
    if args.model == "resnet20":
        global_model = ResNet20(num_classes=6)
    elif args.model == "resnet18":
        global_model = ResNet18(num_classes=6)
    else:
        raise ValueError("Unsupported model selected")

    global_model.to(device)
    global_model.train()

    lr = args.lr
    # make masks
    masks = [make_mask(global_model) for _ in range(args.num_branches)]
    for epoch in range(args.global_rounds):
        if epoch in (15, 45):
            lr = lr / 10
        print(
            "-------------Training Global Epoch:{}/{} with learning rate:{}-------------".format(
                epoch + 1, args.global_rounds, lr
            )
        )
        global_model, masks = train_n_val(
            global_model,
            masks,
            trainloader,
            testloader,
            num_branches=args.num_branches,
            num_epochs=args.local_epochs,
            lr=lr,
            pruning_rate_step=0.1,
            device=device,
        )
        # save model
        save_checkpoint(
            global_model.state_dict(),
            checkpoint=args.checkpoint,
            filename=f"model_{epoch}.pth",
        )
        save_checkpoint(masks, checkpoint=args.checkpoint, filename=f"mask_{epoch}.pth")


def train_n_val(
    global_model,
    masks,
    trainloader,
    testloader,
    num_branches,
    num_epochs,
    device,
    lr=0.01,
    pruning_rate_step=0.1,
):
    criterion = nn.CrossEntropyLoss()
    models = []
    classifier_weight_list = [[] for _ in range(num_branches)]
    val_loss, val_acc = [], []
    step, base = [0 for _ in range(num_branches)], [0.5 for _ in range(num_branches)]
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
                branch_optimizer.zero_grad()
                out = branch_model(inputs)
                temp_target = targets.clone()
                # if args.branches == 2:
                for temp in range(len(temp_target)):
                    if (
                        temp_target[temp] >= idx * 5
                        and temp_target[temp] <= idx * 5 + 4
                    ):
                        temp_target[temp] = temp_target[temp] - idx * 5
                    else:
                        temp_target[temp] = 5
                loss_cb = CB_loss(
                    temp_target,
                    out,
                    [5000, 5000, 5000, 5000, 5000, 250000],
                    6,
                    "focal",
                    0.9999,
                    1.0,
                )
                loss_ce = criterion(out, temp_target)
                loss = loss_ce
                loss.backward()
                branch_optimizer.step()
                _, preds = torch.max(out, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == temp_target)
            epoch_loss = running_loss / len(trainloader.dataset)
            epoch_acc = running_corrects.double() / len(trainloader.dataset)
            print(
                "Training Branch %d | Local Epoch [%d/%d] | Train loss: %.4f | Train Acc: %.4f"
                % (idx, epoch + 1, num_epochs, epoch_loss, epoch_acc)
            )

        models.append(branch_model)
        classifier_weight_list[idx] = branch_model.fc.state_dict()
        # validate branch
        avg_val_loss, avg_val_acc = validate_branch(
            branch_model, testloader, idx, device, grad=False
        )
        val_loss.append(avg_val_loss)
        val_acc.append(avg_val_acc)
        print(
            "Validating Branch %d | Val loss: %.4f | Val Acc: %.4f"
            % (idx, avg_val_loss, avg_val_acc)
        )

        # prune branch mask
        if avg_val_acc > base[idx] and step[idx] < 10:
            for name, param in branch_model.named_parameters():
                if "conv" in name and "weight" in name:
                    prune_mask_layerwise(
                        param.grad.data, param.data, masks[idx][name], pruning_rate_step
                    )
            print(f"Mask for branch:{idx} is pruned by {pruning_rate_step*100}%...")
            step[idx] += 1
            base[idx] += 0.1

    # report global validate loss & accuracy
    global_val_loss_avg = sum(val_loss) / len(val_loss)
    global_val_acc_avg = sum(val_acc) / len(val_acc)
    print(
        f"\nGlobaln Validation Loss avg :{round(global_val_loss_avg, 4)} | Acc avg :{global_val_acc_avg.data}\n"
    )

    # Update weight for global model
    new_global_weights = update_global_model_with_masks(global_model, models, masks)
    global_model.load_state_dict(new_global_weights)

    return global_model, masks


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


if __name__ == "__main__":
    main()
