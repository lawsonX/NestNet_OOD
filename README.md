# NestNet_OOD

## Train
#### Start training a 2 branches resnet18 on CIFAR10
```
CUDA_VISIBLE_DEVICES=0 python src/train.py \
--model resnet18 \
--num_branches 2 \ 
--lr 0.05 \
--base 0.6 \
--step 0.02 \
--local_epochs 2 \
--global_rounds 100 \
--checkpoint checkpoint/resnet18_b2 \
``` 