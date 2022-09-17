import torch


def get_optimizer(cfg, model):
    """
    Gets Adam optimizer
    :param cfg: cfg['train']['optimizer'] part of config
    :param model: resnet-50 model
    :return: optimizer
    """
    print(f'Getting optimizer...')
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    return opt


def get_criterion():
    """
    Gets loss function
    :return: loss function
    """
    print(f'Getting criterion...')
    criterion = torch.nn.CrossEntropyLoss()
    return criterion


def make_training_step(cfg_train, batch, model, criterion, optimizer):
    """
    Makes single parameters updating step.
    :param cfg_train: cfg['train'] part of config
    :param batch: current batch
    :param model: resnet50 model
    :param criterion: criterion
    :param optimizer: optimizer
    :param iter_: current iteration
    :return: current loss value
    """
    images, labels = batch
    images, labels, model = images.cuda(), labels.cuda(), model.cuda()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
