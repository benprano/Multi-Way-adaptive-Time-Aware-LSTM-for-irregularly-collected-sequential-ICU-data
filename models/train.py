import os
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc, average_precision_score
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, mode, path, patience=3, delta=0):
        if mode not in {'min', 'max'}:
            raise ValueError("Argument mode must be one of 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("Argument patience must be a positive integer.")
        if delta < 0:
            raise ValueError("Argument delta must not be a negative number.")

        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0

    def _is_improvement(self, val_score):
        """Return True iff val_score is better than self.best_score."""
        if self.mode == 'max' and val_score > self.best_score + self.delta:
            return True
        elif self.mode == 'min' and val_score < self.best_score - self.delta:
            return True
        return False

    def __call__(self, val_score, model):
        """
        Return True iff self.counter >= self.patience.
        """

        if self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("Val loss improved, Saving model's best weights.")
            return False
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Stopped early. Best val loss: {self.best_score:.4f}')
                return True


class TrainerHelpers:
    def __init__(self, input_dim, hidden_dim, seq_length, output_dim, device, optimizer, loss_criterion, schedulers,
                 num_epochs, patience_n=50, task=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.device = device
        self.optim = optimizer
        self.loss_criterion = loss_criterion
        self.schedulers = schedulers
        self.num_epochs = num_epochs
        self.patience_n = patience_n
        self.task = task

    @staticmethod
    def acc(predicted, label):
        predicted = predicted.sigmoid()
        pred = torch.round(predicted.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train_model(self, model, train_dataloader):
        model.train()
        running_loss, running_corrects = 0.0, 0.0
        for bi, inputs in enumerate(tqdm(train_dataloader, total=len(train_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs, labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            self.optim.zero_grad()
            outputs = model(temporal_features, timestamp, last_data, data_freqs)

            loss = self.loss_criterion(outputs.sigmoid(), labels)
            loss.backward()
            self.optim.step()
            running_loss += loss.item()
            running_corrects += self.acc(outputs, labels)
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects / len(train_dataloader.dataset)
        return epoch_loss, epoch_acc

    def valid_model(self, model, valid_dataloader):
        model.eval()
        running_loss, running_corrects = 0.0, 0.0
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(valid_dataloader, total=len(valid_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs, labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            with torch.no_grad():
                outputs = model(temporal_features, timestamp, last_data, data_freqs)
            loss = self.loss_criterion(outputs.sigmoid(), labels)
            running_loss += loss.item()
            running_corrects += self.acc(outputs, labels)
            fin_targets.append(labels.cpu().detach().numpy())
            fin_outputs.append(outputs.cpu().detach().numpy())
        epoch_loss = running_loss / len(valid_dataloader)
        epoch_accuracy = running_corrects / len(valid_dataloader.dataset)
        return epoch_loss, epoch_accuracy, np.vstack(fin_targets), np.vstack(fin_outputs)

    def eval_model(self, model_class, model_path, test_dataloader):
        # Initialize the model architecture
        model = model_class(self.input_dim, self.hidden_dim, self.seq_length, self.output_dim).to(self.device)
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode
        model.eval()
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(test_dataloader, total=len(test_dataloader), leave=False,
                                         desc='Evaluating on test data')):
            temporal_features, timestamp, last_data, data_freqs, labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            with torch.no_grad():
                outputs = model(temporal_features, timestamp, last_data, data_freqs)

            fin_outputs.append(outputs.sigmoid().cpu().detach().numpy())
            fin_targets.append(labels.cpu().detach().numpy())
        return np.vstack(fin_targets), np.vstack(fin_outputs)

    def _evaluate_model(self, model_class, model_path, test_dataloader):
        targets, predicted, all_decays, fgate_weights = [], [], [], []
        y_pred, y_true = self.eval_model(model_class, model_path, test_dataloader)
        targets.append(y_true)
        predicted.append(y_pred)

        targets_all = [np.vstack(targets[i]) for i in range(len(targets))]
        predicted_all = [np.vstack(predicted[i]) for i in range(len(predicted))]
        return targets_all, predicted_all

    def train_validate_evaluate(self, model_class, model, model_name, train_loader, val_loader, test_loader, model_path):
        best_losses, all_scores, f1_scores_folds = [], [], []
        es = EarlyStopping(mode='min', path=f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                           patience=self.patience_n)
        for epoch in range(self.num_epochs):
            loss, accuracy = self.train_model(model, train_loader)
            eval_loss, eval_accuracy, __, _ = self.valid_model(model, val_loader)
            if self.schedulers is not None:
                self.schedulers.step()
            print(f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, "
                  f"train loss: {loss:.8f}, acc: {accuracy:.8f} | valid loss: {eval_loss:.8f}, acc: {eval_accuracy:.4f}")
            if es(eval_loss, model):
                best_losses.append(es.best_score)
                print("best_score", es.best_score)
                break

        _, _, y_true, y_pred = self.valid_model(model, val_loader)
        print(y_true.shape, y_pred.shape)
        pr_score = average_precision_score(y_true, y_pred)
        print(f"[INFO] PR-AUC ON FOLD :{model_name} -  score val data: {pr_score:.4f}")

        targets, outputs, = self._evaluate_model(model_class, f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                                                 test_loader)

        delta, f1_scr = self.best_threshold(np.vstack(targets), np.vstack(outputs))
        scores = self.metrics_binary(targets, outputs)
        f1_scores_folds.append((delta, f1_scr))
        all_scores.append([scores, f1_scores_folds])

        np.savez(os.path.join(model_path, f"results_data_{model_name}.npz"), auc_pr=scores,
                 true_labels_data=np.vstack(outputs), predicted_labels_data=np.vstack(targets),
                 folds_f1_scores=f1_scores_folds)
        print(f"[INFO] Results on test Folds {all_scores}")

    @staticmethod
    def metrics_binary(targets, predicted):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            fpr, tpr, thresholds = roc_curve(y_pred, y_true)
            auc_score = auc(fpr, tpr)
            pr_score = average_precision_score(y_pred, y_true)
            scores.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return scores

    @staticmethod
    def best_threshold(y_train, train_preds):
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(train_preds, np.array(y_train) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray, rowcount: np.int64, featurecount: np.int64):
        return 1 - (1 - r2_score(actual, predicted)) * (rowcount - 1) / (rowcount - featurecount)

    def metrics_reg(self, targets, predicted, rescale_params):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            target_max, target_min = rescale_params['data_targets_max'], rescale_params['data_targets_min']
            targets_y_true = y_true * (target_max - target_min) + target_min
            targets_y_pred = y_pred * (target_max - target_min) + target_min
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            n = y_true.shape[0]
            r2 = r2_score(targets_y_true, targets_y_pred)
            adj_r2 = self.adjusted_r2(targets_y_true, targets_y_pred, n, self.input_dim)
            scores.append([rmse, mae, r2, adj_r2])
        return scores

    def metrics_reg_imp(self, real, imputed):
        scores = []
        for y_true, y_pred, in zip(real, imputed):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            n = y_true.shape[0]
            adj_r2 = self.adjusted_r2(y_true, y_pred, n, self.input_dim)
            scores.append([rmse, mae, r2, adj_r2])
        return scores
