import time
import numpy as np
import torch

from utils.log_utils import log_metrics


def evaluate(cfg_train, cfg_logging, model, dl, epoch, dataset_type, criterion):
    """
    Evaluates on train/valid data
    :param cfg_eval: cfg['train'] part of config
    :param cfg_logging: cfg['logging'] part of config
    :param model: resnet-50 model
    :param dl: train/valid dataloader
    :param epoch: epoch for logging
    :param dataset_type: type of current data ('train' or 'valid')
    """
    print(f'Evaluating on {dataset_type} data...')
    eval_start_time = time.time()
    correct, total = 0, 0
    losses = []
    model = model.cuda()

    dl_len = len(dl)
    for i, (images, labels) in enumerate(dl):
        images, labels = images.cuda(), labels.cuda()

        if i % 50 == 0:
            print(f'iter: {i}/{dl_len}')

        with torch.no_grad():
            logits = model(images)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += torch.sum(predicted == labels)


        # calculate losses
        loss = criterion(logits, labels)
        losses.append((loss).item())

    log_metrics([f'{dataset_type}_eval/total_loss'], [np.mean(losses)], epoch, cfg_logging)

    accuracy = 100 * correct.item() / total
    print(f'Accuracy on {dataset_type} data: {accuracy}')

    log_metrics([f'{dataset_type}_eval/accuracy'], [accuracy], epoch, cfg_logging)
    print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')
