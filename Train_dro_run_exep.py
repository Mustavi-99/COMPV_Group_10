from dro_utils import *
import os
from dro_data import *
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from dro_train import *


def main():
    reweight_groups = False
    automatic_adjustment = False
    use_normalized_loss = False
    lr = 0.001
    alpha = 0.2
    batch_size = 64
    weight_decay = 0
    gamma = 0.1
    n_epochs = 10
    seed = 0
    generalization_adjustment = "0"
    log_every = 50
    model = "resnet50"
    log_dir = "./logs"
    save_best = True
    save_last = False
    mode = 'w'
    robust = False
    robust_step_size = 0.01
    log_every = 50
    save_step = 10
    save_best = True
    save_last = False

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(os.path.join(log_dir, 'log.txt'), mode=mode)
    set_seed(seed)

    train_data, val_data, test_data = get_confounder_splits(
        data_dir="../datasets/waterbird", augment_data=False)

    loader_kwargs = {'batch_size': batch_size, 'pin_memory': False}
    train_loader = train_data.get_loader(
        train=True, reweight_groups=reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(
        train=False, reweight_groups=False, ** loader_kwargs)
    test_loader = test_data.get_loader(
        train=False, reweight_groups=False, **loader_kwargs)

    print(
        f'Train data: {train_data.__len__()} | Val data:  {val_data.__len__()} | Test data:  {test_data.__len__()}')

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)

    logger.flush()

    criterion = nn.CrossEntropyLoss(reduction='none')

    train_csv_logger = CSVBatchLogger(os.path.join(
        log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger = CSVBatchLogger(os.path.join(
        log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger = CSVBatchLogger(os.path.join(
        log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    train(
        model=model,
        criterion=criterion,
        dataset=data,
        logger=logger,
        train_csv_logger=train_csv_logger,
        val_csv_logger=val_csv_logger,
        test_csv_logger=test_csv_logger,
        alpha=alpha,
        gamma=gamma,
        robust=robust,
        robust_step_size=robust_step_size,
        use_normalized_loss=use_normalized_loss,
        generalization_adjustment=generalization_adjustment,
        lr=lr,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        log_every=log_every,
        save_step=save_step,
        save_last=save_last,
        save_best=save_best,
        log_dir=log_dir,
        reweight_groups=reweight_groups,
        automatic_adjustment=automatic_adjustment)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()


if __name__ == "__main__":
    main()
