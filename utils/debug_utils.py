import torchvision
import torch
import numpy as np


def overfit_on_batch(cfg_overfit_on_batch, cfg_train, train_dl, model, optimizer, criterion):
    """
    Overfits on one batch
    :param cfg_overfit_on_batch: cfg['debug']['overfit_on_batch'] part of config
    :param train_dl: train dataloader
    :param model: resnet50 model
    :param optimizer: optimizer
    :param criterion: criterion
    """
    train_dl = iter(train_dl)
    images, labels = next(train_dl)
    model = model.cuda()
    accuracies = []

    for iter_ in range(cfg_overfit_on_batch['nb_iters']):
        optimizer.zero_grad()
        logits = model(images.cuda()).cpu()
        loss = criterion(logits, labels)
        _, predicted = torch.max(logits.data, 1)
        accuracy = torch.sum(predicted == labels).item() / labels.size(0) * 100
        print(f'iter: {iter_}, acc: {accuracy}, loss: {loss.item()}')

        accuracies.append(accuracy)
        if len(accuracies) >= 5 and np.min(accuracies[-5:]) == 100:
            break

        loss.backward()
        optimizer.step()
    print(f'Overfitting on batch is finished.')


def save_batch_images(cfg, train_dl, valid_dl):
    """
    Saves several batches of images as .png file
    :param cfg: cfg['debug']['save_batch'] part of config
    :param train_dl: train dataloader to saves batches from
    :param valid_dl: valid dataloader to saves batches from
    """
    for dl in [train_dl, valid_dl]:
        dataset_type = dl.dataset.dataset_type
        print(dataset_type)
        dl = iter(dl)
        for i in range(cfg['nrof_batches_to_save']):
            images, labels = next(dl)
            print(f'batch {i} labels: {labels}')
            torchvision.utils.save_image(images, cfg['path_to_save'] + f'{dataset_type}_batch_{i}.png')
