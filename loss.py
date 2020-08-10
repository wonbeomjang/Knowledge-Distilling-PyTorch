import torch.nn.functional as F


def loss_kd(preds, labels, teacher_preds, params):
    T = params.temperature
    alpha = params.alpha
    loss = F.kl_div(F.log_softmax(preds / T, dim=1), F.softmax(teacher_preds / T, dim=1),
                    reduction='batchmean') * T * T * alpha + F.cross_entropy(preds, labels) * (1. - alpha)
    return loss
