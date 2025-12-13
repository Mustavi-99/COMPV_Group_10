# dataloader_cifar.py
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset):
    """
    Minimal-change CIFAR dataset compatible with DivideMix pipeline.
    - Maintains original modes: 'all', 'labeled', 'unlabeled', 'test'
    - For CIFAR-10: maps labels to binary bird-vs-rest:
        binary_label = 1 if original_label == 2 else 0
      and stores original multiclass labels in self.original_label for analysis.
    - For other datasets (e.g., cifar100) behavior is unchanged (no binary mapping).
    """
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=None, probability=None, log=None):
        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.dataset = dataset
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}  # asymmetric noise mapping

  
        self.train_data = None
        self.test_data = None
        self.train_label = None 
        self.noise_label = None 
        self.binary_label = None 
        self.original_label = None 

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                # original multiclass labels
                orig_test_label = np.array(test_dic['labels'])
                # store original labels for external access
                self.original_label = orig_test_label
                # binary mapping only for cifar10
                self.test_label = (orig_test_label == 2).astype(np.int64)
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                orig_test_label = np.array(test_dic['fine_labels'])
                self.original_label = orig_test_label
                # For cifar100 we do not create a binary mapping by default
                self.test_label = orig_test_label
        else:
            # TRAIN / other modes
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            # Load or inject noise (original multiclass noisy labels)
            if os.path.exists(noise_file) and len(noise_file) > 0:
                noise_label = json.load(open(noise_file, "r"))
                noise_label = np.array(noise_label).astype(int).tolist()
            else:
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                            noise_label.append(int(noiselabel))
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(int(noiselabel))
                    else:
                        noise_label.append(int(train_label[i]))
                if len(noise_file) > 0:
                    print("save noisy labels to %s ..." % noise_file)
                    json.dump(noise_label, open(noise_file, "w"))

            # At this point:
            # - train_label contains the clean (original) labels 0..9
            # - noise_label contains the noisy original labels 0..9
            train_label = np.array(train_label).astype(int)
            noise_label = np.array(noise_label).astype(int)
            
            # Store original multiclass labels for later analysis (accessible as dataset.original_label)
            self.original_label = train_label.copy()

            # Create binary mapping ONLY for CIFAR-10
            if self.dataset == 'cifar10':
                # binary: bird (class 2) -> 1, others -> 0
                binary_noisy = (noise_label == 2).astype(int)
                binary_clean = (train_label == 2).astype(int)
            else:
                # For other datasets we keep labels as-is (no binary mapping)
                binary_noisy = noise_label.copy()
                binary_clean = train_label.copy()

            # Attach data depending on mode
            if self.mode == 'all':
                self.train_data = train_data
                # store original noisy labels (multiclass)
                self.noise_label = noise_label.tolist()
                # binary labels used by DivideMix/training
                self.binary_label = binary_noisy.tolist()
                # print(len(self.noise_label),len(self.binary_label))
            else:
                # labeled / unlabeled modes use pred mask
                # pred is expected as a numpy array-like of 0/1 values where 1 -> labeled
                if pred is None:
                    raise ValueError("pred must be provided for mode 'labeled' or 'unlabeled'")

                # pred might be a torch tensor or numpy array
                pred_arr = np.array(pred)
                # labeled indices (pred == 1)
                if self.mode == "labeled":
                    pred_idx = pred_arr.nonzero()[0]
                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred_arr).nonzero()[0]
                else:
                    raise ValueError("Unknown mode: %s" % self.mode)

                # Probability is only meaningful for labeled mode
                if self.mode == "labeled":
                    # extract corresponding probability entries if provided
                    if probability is None:
                        raise ValueError("probability must be provided for mode 'labeled'")
                    prob_list = np.array([probability[i] for i in pred_idx])
                    self.probability = prob_list

                    # compute AUC of predicted-cleanness vs actual cleanliness (clean = binary_clean)
                    clean_mask = (binary_noisy == binary_clean)  # whether noisy label equals clean label in binary sense
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    # probability is per-sample cleaned probability (higher means more likely clean)
                    auc_meter.add(prob_list, clean_mask[pred_idx].astype(int))
                    try:
                        auc, _, _ = auc_meter.value()
                    except Exception:
                        auc = float('nan')
                    if log is not None:
                        log.write('Number of labeled samples:%d   AUC:%.3f\n' % (pred_arr.sum(), auc))
                        log.flush()
                else:
                    self.probability = None

                self.train_data = train_data[pred_idx]
                # store multiclass noisy labels for the selected subset
                self.noise_label = noise_label[pred_idx].tolist()
                # binary labels for training
                self.binary_label = binary_noisy[pred_idx].tolist()
                self.original_label = train_label[pred_idx].copy()
                print("%s data has a size of %d" % (self.mode, len(self.binary_label)))

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_np, target_binary, prob = self.train_data[index], self.binary_label[index], self.probability[index]
            # For compatibility, target returned to training code is binary (0/1)
            img = Image.fromarray(img_np)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, int(target_binary), float(prob)
        elif self.mode == 'unlabeled':
            img_np = self.train_data[index]
            img = Image.fromarray(img_np)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img_np, target_binary = self.train_data[index], self.binary_label[index]
            img = Image.fromarray(img_np)
            img = self.transform(img)
            # Return (img, target_binary, index) to remain compatible with original code.
            return img, int(target_binary), index
        elif self.mode == 'test':
            img_np, target_binary = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img_np)
            img = self.transform(img)
            # For test mode we return binary label (used by evaluation scripts).
            # Original multiclass label is available in dataset.original_label externally.
            return img, int(target_binary)

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    """
    Minimal-change dataloader wrapper for DivideMix usage.
    Usage:
        loader = cifar_dataloader('cifar10', r, noise_mode, batch_size, num_workers, root_dir, log, noise_file)
        warmup_loader = loader.run('warmup')
        labeled_loader, unlabeled_loader = loader.run('train', pred=pred, prob=prob)
        test_loader = loader.run('test')
    """
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, pred=None, prob=None):
        """
        mode:
          - 'warmup' : return DataLoader over the entire training set (mode='all')
          - 'train' : return (labeled_trainloader, unlabeled_trainloader) using pred & prob
          - 'test'  : return test_loader
          - 'eval_train' : return evaluation loader for the FULL train set (mode='all', no augmentation)
        """
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                       root_dir=self.root_dir, transform=self.transform_train,
                                       mode="all", noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            if pred is None or prob is None:
                raise ValueError("pred and prob must be provided for mode 'train'")

            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train,
                                            mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob, log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train,
                                              mode="unlabeled", noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        else:
            raise ValueError("Unknown mode: %s" % mode)
