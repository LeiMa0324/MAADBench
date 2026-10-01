import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from benchmark_AD.methods.anomaly_transformer.utils import *
from benchmark_AD.methods.anomaly_transformer.model.AnomalyTransformer import AnomalyTransformer
from benchmark_AD.methods.anomaly_transformer.data_loader import get_loader_segment
from benchmark_AD.utils import log_results, predict_scores


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            # self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            # self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    # def save_checkpoint(self, val_loss, val_loss2, model, path):
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
    #     self.val_loss_min = val_loss
    #     self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config, train_data=None, train_labels=None, test_data=None, test_labels=None, args=None):

        self.__dict__.update(Solver.DEFAULTS, **config)
        self.args = args

        loader_kwargs = dict(
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=1,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels
        )

        if np.asarray(train_data).ndim == 3:
            # Caller already padded per-trace windows; do not segment the trace axis.
            from torch.utils.data import DataLoader, TensorDataset
            self.train_loader = DataLoader(TensorDataset(torch.as_tensor(train_data), torch.as_tensor(train_labels)), batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(TensorDataset(torch.as_tensor(test_data), torch.as_tensor(test_labels)), batch_size=self.batch_size, shuffle=False)
        else:
            self.train_loader = get_loader_segment(**loader_kwargs, mode='train')
            self.test_loader = get_loader_segment(**loader_kwargs, mode='test')
        
        # self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
        #                                        mode='train',
        #                                        dataset=self.dataset)
        # self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
        #                                       mode='val',
        #                                       dataset=self.dataset)
        # self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
        #                                       mode='test',
        #                                       dataset=self.dataset)
        # self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
        #                                       mode='thre',
        #                                       dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self, original_lengths_test, train_ids, test_ids):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        # path = self.model_save_path
        # if not os.path.exists(path):
        #     os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            # evaluate the model 
            self.test(epoch, self.num_epochs, original_lengths=original_lengths_test, train_ids=train_ids, test_ids=test_ids)

            # print(
            #     "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
            #         epoch + 1, train_steps, train_loss, vali_loss1))
            # early_stopping(vali_loss1, vali_loss2, self.model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self, epoch, epoch_total, original_lengths=None, train_ids=None, test_ids=None):
        # self.model.load_state_dict(
        #     torch.load(
        #         os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        # attens_energy = []
        # for i, (input_data, labels) in enumerate(self.train_loader):
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)
        #     loss = torch.mean(criterion(input, output), dim=-1)
        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if u == 0:
        #             series_loss = my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss = my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #         else:
        #             series_loss += my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss += my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature

        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(test_energy)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        # print("Threshold :", thresh)

        mask = np.concatenate([np.concatenate([np.ones(length), np.zeros(100 - length)]) for length in original_lengths])  # remove padding from test set
        test_energy = test_energy[mask == 1]
        test_labels = test_labels[mask == 1]



        # (3) evaluation on the test set
        # test_labels = []
        # attens_energy = []
        # for i, (input_data, labels) in enumerate(self.thre_loader):
        #     input = input_data.float().to(self.device)
        #     output, series, prior, _ = self.model(input)

        #     loss = torch.mean(criterion(input, output), dim=-1)

        #     series_loss = 0.0
        #     prior_loss = 0.0
        #     for u in range(len(prior)):
        #         if u == 0:
        #             series_loss = my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss = my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #         else:
        #             series_loss += my_kl_loss(series[u], (
        #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                            self.win_size)).detach()) * temperature
        #             prior_loss += my_kl_loss(
        #                 (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                         self.win_size)),
        #                 series[u].detach()) * temperature
        #     metric = torch.softmax((-series_loss - prior_loss), dim=-1)

        #     cri = metric * loss
        #     cri = cri.detach().cpu().numpy()
        #     attens_energy.append(cri)
        #     test_labels.append(labels)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)
        # test_labels = np.array(test_labels)

        pred = predict_scores(self.args, test_energy, test_labels)

        gt = test_labels

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        f1 = f1_score(gt, pred)
        acc = accuracy_score(gt, pred)
        auc = roc_auc_score(gt, test_energy)
        bal_acc = balanced_accuracy_score(gt, pred)

        # print metrics
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")  

        log_results(self.args, epoch, epoch_total, f1, acc, auc, bal_acc, model=self.model, gt_labels=gt, pred_labels=pred, pred_scores=test_energy, train_ids=train_ids, test_ids=test_ids)
        
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        # anomaly_state = False
        # for i in range(len(gt)):
        #     if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        #         anomaly_state = True
        #         for j in range(i, 0, -1):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #         for j in range(i, len(gt)):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #     elif gt[i] == 0:
        #         anomaly_state = False
        #     if anomaly_state:
        #         pred[i] = 1

        # pred = np.array(pred)
        # gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)

        # from sklearn.metrics import precision_recall_fscore_support
        # from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
        #                                                                       average='binary')
        # print(
        #     "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))