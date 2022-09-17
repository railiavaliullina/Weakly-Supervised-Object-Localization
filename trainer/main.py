import torch
import time
import numpy as np

from models.resnet50 import get_model
from data.coco_80_dataset import get_data
from utils.train_utils import get_optimizer, get_criterion, make_training_step
from utils.eval_utils import evaluate
from utils.debug_utils import save_batch_images, overfit_on_batch
from utils.log_utils import start_logging, end_logging, log_metrics, log_params
from configs.config import cfg


def train(cfg, train_dl, test_dl, model, optimizer, criterion):
    # check data before training
    if cfg['debug']['save_batch']['enable']:
        save_batch_images(cfg['debug']['save_batch'], train_dl, test_dl)

    # check training procedure before training
    if cfg['debug']['overfit_on_batch']['enable']:
        overfit_on_batch(cfg['debug']['overfit_on_batch'], cfg['train'], train_dl, model, optimizer, criterion)

    # save experiment name and experiment params to mlflow
    start_logging(cfg['logging'], experiment_name='baseline')
    log_params(cfg['logging'])

    global_step, start_epoch = 0, 0
    if cfg['logging']['load_model']:
        print(f'Trying to load checkpoint from epoch {cfg["logging"]["epoch_to_load"]}...')
        checkpoint = torch.load(cfg['logging']['checkpoints_dir'] + f'checkpoint_{cfg["logging"]["epoch_to_load"]}.pth')
        load_state_dict = checkpoint['model']
        model.load_state_dict(load_state_dict)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        print(f'Successfully loaded checkpoint from epoch {cfg["logging"]["epoch_to_load"]}.')

    # evaluate on train and test data before training
    if cfg['eval']['evaluate_before_training']:
        model.eval()
        if cfg['eval']['evaluate_on_train_data']:
            evaluate(cfg['train'], cfg['logging'], model, train_dl, -1, 'train', criterion)
        evaluate(cfg['train'], cfg['logging'], model, test_dl, -1, 'valid', criterion)
        model.train()

    nb_iters_per_epoch = len(train_dl.dataset) // train_dl.batch_size

    # training loop
    for epoch in range(start_epoch, cfg['train']['epochs']):
        losses = []
        epoch_start_time = time.time()
        print(f'Epoch: {epoch}')
        for iter_, batch in enumerate(train_dl):
            loss = make_training_step(cfg['train'], batch, model, criterion, optimizer)
            losses.append(loss)
            global_step += 1

            log_metrics(['train/loss'], [loss], global_step, cfg['logging'])

            if global_step % 100 == 0:
                mean_loss = np.mean(losses[:-20]) if len(losses) > 20 else np.mean(losses)
                print(f'step: {global_step}, total_loss: {mean_loss}')

        # log mean loss per epoch
        log_metrics(['train/mean_loss'], [np.mean(losses[:-nb_iters_per_epoch])], epoch, cfg['logging'])
        print(f'Epoch time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        # save model
        if cfg['logging']['save_model'] and epoch % cfg['logging']['save_frequency'] == 0:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'opt': optimizer.state_dict(),
            }
            torch.save(state, cfg['logging']['checkpoints_dir'] + f'checkpoint_{epoch}.pth')

        # evaluate on train and test data
        model.eval()
        if cfg['eval']['evaluate_on_train_data']:
            evaluate(cfg['train'], cfg['logging'], model, train_dl, epoch, 'train', criterion)
        evaluate(cfg['train'], cfg['logging'], model, test_dl, epoch, 'valid', criterion)
        model.train()

    end_logging(cfg['logging'])


def run(cfg):
    train_dl, test_dl = get_data(cfg['data'])
    model = get_model(cfg)
    optimizer = get_optimizer(cfg['train']['optimizer'], model)
    criterion = get_criterion()

    train(cfg, train_dl, test_dl, model, optimizer, criterion)


if __name__ == '__main__':
    start_time = time.time()
    run(cfg)
    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')
