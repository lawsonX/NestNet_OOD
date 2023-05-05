import os
import copy
import torch
import numpy as np

def global_pruning_v2(masks, grads, models, additional_pruning_rate):
    """
    这个版本是想实现 global channel pruning
    TODO:这个函数需要debug,
    问题:1. 输出mask貌似没有剪枝成功 2.运行速度特别慢,算很久,怀疑是使用numpy造成的。

    Args:
        masks (list): list(mask * num_branches)
        grads (dict): 梯度
        models (list): list(model * num_branches)
        additional_pruning_rate (float): default 0.1 ,每次调用这个函数会累加0.1的pruning rate

    Returns:
        list: list(mask * num_branches)
    """
    
    weights = []
    for model in models:
        conv_weights = {k: v for k, v in model.state_dict().items() if 'conv' in k}
        weights.append(conv_weights)
        
    assert masks[0].keys() == grads[0].keys() == weights[0].keys()
    
    scores = []
    all_layer_indices = []

    # Calculate the number of already pruned weights
    num_pruned_weights = sum([torch.sum(torch.tensor(mask[mask_elem]).eq(0).int()).item() for mask in masks for mask_elem in mask])
    total_weights = sum([sum([np.prod(user_masks[mask].shape) for mask in user_masks]) for user_masks in masks])

    # Calculate the number of weights that need to be pruned to achieve the desired pruning rate
    desired_pruned_weights = int(total_weights * additional_pruning_rate)
    remaining_pruned_weights = max(desired_pruned_weights - num_pruned_weights, 0)

    # Adjust the global pruning rate based on the remaining weights to be pruned
    adjusted_pruning_rate = remaining_pruned_weights / (total_weights - num_pruned_weights)

    # Calculate scores for all layers and branches
    for idx, (user_mask, user_grad, user_weight) in enumerate(zip(masks, grads, weights)):
        user_layer_indices = []
        user_scores = []
        for mask, grad, weight in zip(user_mask.values(), user_grad.values(), user_weight.values()):
            # weight = weight.reshape(grad.shape)
            grad = grad.cuda()
            score = weight * grad
            layer_indices = torch.argsort(score.view(-1)).tolist()
            user_layer_indices.append(layer_indices)
            user_scores.extend([score.view(-1)[index].item() for index in layer_indices]) # 修改这一行
        all_layer_indices.append(user_layer_indices)
        scores.append(user_scores)
    # Calculate the global pruning threshold and number of elements to prune
    num_pruning = int(len(scores) * adjusted_pruning_rate)
    threshold = np.percentile(scores, adjusted_pruning_rate * 100)

    # Prune the masks based on the global threshold and num_pruning
    pruned_count = 0
    for idx, (user_mask, user_layer_indices) in enumerate(zip(masks, all_layer_indices)):
        if pruned_count >= num_pruning:
            break

        for mask, layer_indices in zip(user_mask, user_layer_indices):
            indices_to_prune = [i for i, score in zip(layer_indices, scores[idx]) if score <= threshold]
            pruned_indices = indices_to_prune[:min(num_pruning - pruned_count, len(indices_to_prune))]
            mask[pruned_indices] = 0

            pruned_count += len(pruned_indices)

    return masks

def global_pruning_v1(masks, mats_score, pruning_ratio):
    """
    这个版本实现了global pruning, 
    剪枝的最小单位是single weight, 并非 channel

    Args:
        masks (_type_): _description_
        mats_score (dict): 输出的是score, 不同于v2,在train.py: line264&265 有说明
        pruning_ratio (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Step 1: Flatten all masks and concatenate them into a single vector
    flattened_masks = [mask[key].flatten() for mask in masks for key in mask.keys()]
    flattened_masks_concat = torch.cat(flattened_masks)
    
    flattened_score = [score[key].flatten() for score in mats_score for key in score.keys()]
    flattened_score_concat = torch.cat(flattened_score)
    
    flattened_score_concat, i = torch.sort(flattened_score_concat)
    num_pruning = int(flattened_score_concat.numel() * pruning_ratio)
    flattened_masks_concat[i[:num_pruning]] = 0

    # Step 3: Update all masks, setting pruned place to 0
    mask_index = 0
    for mask_dict in masks:
        for key in mask_dict.keys():
            mask_values = flattened_masks_concat[mask_index:mask_index + mask_dict[key].numel()]
            mask_dict[key] = mask_values.view(mask_dict[key].shape).float()
            mask_index += mask_dict[key].numel()

    return masks

def update_global_model_v2(global_model, branch_models, masks, alpha=0.1):
    """ 
    负责全局 model的权重更新函数
    聚合各分的conv.weight, 根据被选择的次数平均,如果都没有选择(mask=0),weight就不变化
    conv.weight以外的权重直接就和平均
    

    Args:
        global_model (Class): Class, a complete single model, 
        branch_models (list): list(model * num_branches)
        masks (list): list(mask * num_branches)
        device (_type_): device
        alpha (float): 更新比例. Defaults to 0.1.

    Returns:
        class: Weight updated Single model
    """
    global_weights = global_model.state_dict()
    init_weights = copy.deepcopy(global_model.state_dict())

    mask_count = {}
    for i, params in enumerate(zip(*[m for m in masks])):
        sum_mask = torch.zeros_like(masks[0][params[0]])
        for b, m, p in zip(branch_models, masks, params):
            sum_mask += m[p]
        mask_count[params[0]] = sum_mask

    sum_state_dict = branch_models[0].state_dict()
    for model in branch_models[1:]:
        for key in model.state_dict():
            sum_state_dict[key] += model.state_dict()[key]

    for key in global_weights.keys():
        if "conv" in key and "weight" in key:
            global_weights[key] = torch.where(
                mask_count[key] == 0,
                init_weights[key],
                sum_state_dict[key] / mask_count[key],
            )
        else:
            global_weights[key] = sum_state_dict[key] / len(branch_models)
        
        updated_weights = (1 - alpha) * global_weights + alpha * global_weights
        global_model.state_dict()[key].data.copy_(updated_weights)

    return global_model

def update_global_model_v1(global_model, branch_models, masks, device, alpha=0.1):

    for key in global_model.state_dict().keys():
        global_weights = global_model.state_dict()[key]
        branch_weights_sum = torch.zeros_like(global_weights).to(device).float()
        mask_counts = torch.zeros_like(global_weights).to(device)
        for branch_idx, branch_model in enumerate(branch_models):
            if "conv" in key:
                branch_weights = branch_model.state_dict()[key]
                mask = masks[branch_idx][key]
                mask = mask.to(device)
                branch_weights = branch_weights * mask
                mask_counts += mask
                branch_weights_sum += branch_weights

        branch_weight_avg = torch.where(mask_counts == 0, torch.zeros_like(branch_weights_sum), branch_weights_sum / mask_counts)  
        updated_weights = (1 - alpha) * global_weights + alpha * branch_weight_avg
        global_model.state_dict()[key].data.copy_(updated_weights)
 
    return global_model

def create_masked_branch_model(global_model, classifier_weight, mask):
    branch_model = copy.deepcopy(global_model)
    for name, param in branch_model.named_parameters():
        if name in mask:
            param.data.mul_(mask[name])
    if len(classifier_weight) != 0:  # 分支模型加载自己的输出层
        branch_model.fc.load_state_dict(classifier_weight)
    return branch_model

def make_mask(model):
    mask = {}
    for name, param in model.named_parameters():
        if "weight" in name and "conv" in name:
            mask[name] = torch.ones_like(param.data)
    return mask

def make_grad_mat(model):
    mat = {}
    for name, param in model.named_parameters():
        if "weight" in name and "conv" in name:
            mat[name] = torch.zeros_like(param.data)
    return mat

def compute_sparsity(mask):
    total_weight_elements = 0
    weight_sum = 0
    for key, value in mask.items():
        total_weight_elements += torch.numel(value)
        weight_sum += torch.sum(value)
    remain_ratio = weight_sum / total_weight_elements
    return remain_ratio

def compute_score(grad, weight):
    score = weight * grad
    score = torch.sum(score, dim=tuple(range(1, len(score.shape))))
    return score

def prune_mask_layerwise(grad, weight, mask, pruning_rate):
    score = weight * grad
    score = torch.sum(score, dim=tuple(range(1, len(score.shape))))
    score, i = torch.sort(score)
    num_pruning = int(score.numel() * pruning_rate)
    mask[i[:num_pruning]] = 0


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)