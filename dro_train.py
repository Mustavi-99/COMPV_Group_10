from dro_loss import LossComputer
from dro_utils import get_device
import numpy as np
from tqdm import tqdm
import torch
import os


def run_epoch(epoch,
              model,
              optimizer,
              loader,
              loss_computer,
              logger,
              csv_logger,
              weight_decay,
              is_training,
              log_every=50):

    device = get_device()
    if is_training:
        model.train()
    else:
        model.eval()

    prog_bar_loader = tqdm(loader)

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]

            outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx,
                               loss_computer.get_stats(model, weight_decay))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

            # remove this while training
            # if batch_idx == 1:
            #     break

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx,
                           loss_computer.get_stats(model, weight_decay))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model,
          criterion,
          dataset,
          logger,
          train_csv_logger,
          val_csv_logger,
          test_csv_logger,
          robust,
          alpha,
          gamma,
          robust_step_size,
          use_normalized_loss,
          generalization_adjustment,
          lr,
          weight_decay,
          n_epochs,
          log_every,
          save_step,
          save_last,
          save_best,
          log_dir,
          reweight_groups,
          automatic_adjustment,
          btl=False,
          minimum_variational_weight=0):
    device = get_device()
    model = model.to(device)

    adjustments = [float(c) for c in generalization_adjustment.split(',')]
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=robust,
        dataset=dataset['train_data'],
        alpha=alpha,
        gamma=gamma,
        adj=adjustments,
        step_size=robust_step_size,
        normalize_loss=use_normalized_loss,
        btl=btl,
        min_var_weight=minimum_variational_weight)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    best_val_acc = 0
    for epoch in range(n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loader=dataset['train_loader'],
            loss_computer=train_loss_computer,
            logger=logger,
            csv_logger=train_csv_logger,
            weight_decay=weight_decay,
            is_training=True,
            log_every=log_every
        )

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=robust,
            dataset=dataset['val_data'],
            step_size=robust_step_size,
            alpha=alpha)

        run_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            loader=dataset['val_loader'],
            loss_computer=val_loss_computer,
            logger=logger,
            csv_logger=val_csv_logger,
            weight_decay=weight_decay,
            is_training=False)

        logger.write(f'\nTest:\n')
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=robust,
                dataset=dataset['test_data'],
                step_size=robust_step_size,
                alpha=alpha)
            run_epoch(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                loader=dataset['test_loader'],
                loss_computer=test_loss_computer,
                logger=logger,
                csv_logger=test_csv_logger,
                weight_decay=weight_decay,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if epoch % save_step == 0:
            torch.save(model, os.path.join(log_dir, '%d_model.pth' % epoch))

        if save_last:
            torch.save(model, os.path.join(log_dir, 'last_model.pth'))

        if save_best:
            if robust or reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * \
                torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
