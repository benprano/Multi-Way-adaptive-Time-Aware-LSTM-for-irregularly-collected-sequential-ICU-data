import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class PrepareData:
    def __init__(self, batch_size, n_folds=2, test_size=0.2, task=True):
        super(PrepareData, self).__init__()
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.test_size = test_size
        self.task = task
        self.test_data_loader = []
        self.train_data_loader = []
        self.valid_data_loader = []
        self.all_test_notes = []
        if self.task:
            self.kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.n_folds * 3)
            self.sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
        else:
            self.kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.n_folds * 3)
            self.ts = ShuffleSplit(n_splits=1, test_size=self.test_size)

    def double_cv(self, train_x, train_t, train_last, train_freq, train_y,
                  test_x, test_t, test_last, test_freq, test_y):

        def collate_fn(batch):
            return tuple(zip(*batch))

        train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32),
                                      torch.tensor(train_t, dtype=torch.float32),
                                      torch.tensor(train_last, dtype=torch.float32),
                                      torch.tensor(train_freq, dtype=torch.float32),
                                      torch.tensor(train_y, dtype=torch.float32))

        valid_dataset = TensorDataset(torch.tensor(test_x, dtype=torch.float32),
                                      torch.tensor(test_t, dtype=torch.float32),
                                      torch.tensor(test_last, dtype=torch.float32),
                                      torch.tensor(test_freq, dtype=torch.float32),
                                      torch.tensor(test_y, dtype=torch.float32))

        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=4)
        valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size,
                                       shuffle=False, num_workers=4)

        return train_data_loader, valid_data_loader

    def cv_folds_split(self, args):
        nb_train_size = 0
        data, notes, time_data, last_data, features_freqs, target = args
        for index, (train_index, test_index) in enumerate(self.kfold.split(data, target)):
            x_train, x_test = data[train_index], data[test_index]
            _, x_test_notes = notes[train_index], notes[test_index]
            x_train_last, x_test_last = last_data[train_index], last_data[test_index]
            x_train_freq, x_test_freq = features_freqs[train_index], features_freqs[test_index]
            x_train_t, x_test_t = time_data[train_index], time_data[test_index]
            y_train, y_test = target[train_index], target[test_index]

            _, test_loader = self.double_cv(x_train, x_train_t, x_train_last, x_train_freq, y_train,
                                            x_test, x_test_t, x_test_last, x_test_freq, y_test)
            # Splitting Training helpers into Training and Validation helpers
            if self.task:
                self.sss.get_n_splits(x_train, y_train)
                train_in_index, val_index = next(self.sss.split(x_train, y_train))
                # Training data
                x_train_in, x_val_in = x_train[train_in_index], x_train[val_index]
                x_train_in_t, x_val_in_t = x_train_t[train_in_index], x_train_t[val_index]
                x_train_in_last, x_val_in_last = x_train_last[train_in_index], x_train_last[val_index]
                x_train_in_freq, x_val_in_freq = x_train_freq[train_in_index], x_train_freq[val_index]
                y_train_fold, y_valid_fold = y_train[train_in_index], y_train[val_index]
                train_loader, val_loader = self.double_cv(x_train_in, x_train_in_t, x_train_in_last, x_train_in_freq,
                                                          y_train_fold,
                                                          x_val_in, x_val_in_t, x_val_in_last, x_val_in_freq,
                                                          y_valid_fold)

                self.train_data_loader.append(train_loader)
                self.valid_data_loader.append(val_loader)

                self.all_test_notes.append(x_test_notes)
                self.test_data_loader.append(test_loader)
                nb_train_size += x_train_in.shape[0]
            else:
                self.ts.get_n_splits(x_train, y_train)
                train_in_index, val_index = next(self.ts.split(x_train, y_train))
                # Training data
                x_train_in, x_val_in = x_train[train_in_index], x_train[val_index]
                x_train_in_t, x_val_in_t = x_train_t[train_in_index], x_train_t[val_index]
                x_train_in_last, x_val_in_last = x_train_last[train_in_index], x_train_last[val_index]
                x_train_in_freq, x_val_in_freq = x_train_freq[train_in_index], x_train_freq[val_index]
                y_train_fold, y_valid_fold = y_train[train_in_index], y_train[val_index]
                train_loader, val_loader = self.double_cv(x_train_in, x_train_in_t, x_train_in_last, x_train_in_freq,
                                                          y_train_fold,
                                                          x_val_in, x_val_in_t, x_val_in_last, x_val_in_freq,
                                                          y_valid_fold)

                self.train_data_loader.append(train_loader)
                self.valid_data_loader.append(val_loader)

                self.all_test_notes.append(x_test_notes)
                self.test_data_loader.append(test_loader)
                nb_train_size += x_train_in.shape[0]
        return self.test_data_loader, self.train_data_loader, self.valid_data_loader, self.all_test_notes, nb_train_size
