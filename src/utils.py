import os
import copy
import torch

def global_pruning(mask_list, mats_score, pruning_ratio):
    # Step 1: Flatten all masks and concatenate them into a single vector
    flattened_masks = [mask[key].flatten() for mask in mask_list for key in mask.keys()]
    flattened_masks_concat = torch.cat(flattened_masks)
    
    flattened_score = [score[key].flatten() for score in mats_score for key in score.keys()]
    flattened_score_concat = torch.cat(flattened_score)
    
    flattened_score_concat, i = torch.sort(flattened_score_concat)
    num_pruning = int(flattened_score_concat.numel() * pruning_ratio)
    flattened_masks_concat[i[:num_pruning]] = 0

    # Step 3: Update all masks, setting pruned channels to 0
    mask_index = 0
    for mask_dict in mask_list:
        for key in mask_dict.keys():
            mask_values = flattened_masks_concat[mask_index:mask_index + mask_dict[key].numel()]
            mask_dict[key] = mask_values.view(mask_dict[key].shape).float()
            mask_index += mask_dict[key].numel()

    return mask_list

def update_global_model_with_masks(global_model, branch_models, masks):
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

    return global_weights


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