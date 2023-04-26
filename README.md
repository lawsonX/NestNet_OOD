# NestNet_OOD

## Train
#### Start training a 2 branches resnet18 on CIFAR10
```
CUDA_VISIBLE_DEVICES=0 python src/train.py \
--model resnet18 \
--num_branches 2 \ 
--lr 0.05 \
--schedule 15 45 70 \
--prune_mode global \
--base 0.5 \
--step 0.02 \
--local_epochs 2 \
--global_rounds 90 \
--checkpoint checkpoint/global_prune/0426/res18_prune_b2 \
``` 