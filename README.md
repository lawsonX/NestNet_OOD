# NestNet_OOD

## History
目前主要可能存在问题的函数是：
```
util/global_pruning_v2
util/global_pruning_v1
util/update_global_model_v2
util/update_global_model_v1
具体情况在每个函数里写了注释

```

## Usage
#### Start training a 2 branches resnet18 on CIFAR10
```
CUDA_VISIBLE_DEVICES=0 python src/train.py \
--model resnet18 \
--num_branches 2 \ 
--lr 0.01 \
--local_epochs 4 \
--global_rounds 40 \
--schedule 12 30 \
--prune_mode global \
--base 0.7 \
--step 0.02 \
--prune-rate 0.05 \
--checkpoint checkpoint/exp1
``` 