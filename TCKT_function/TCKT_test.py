import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # 0, 1, 2, ...


import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import tqdm
import time
from utils.utils import logger_fn

from .meta import KTM
from .TCKT_Net_test_best_global_dict1 import TCKTNet, py_name

print("tckt_test_best_global_dict")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

py_name = py_name
logger = logger_fn("tflog", "logs/{0}/training-{1}.log".format(py_name, time.asctime()).replace(':', '_'))

def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def compute_diff(e_data, a_data):

    problem = []
    answer = []
    [rows, cols] = e_data.shape
    for i in range(rows):
        for j in range(cols):
            if e_data[i][j] == 0:
                break
            else:
                problem.append(e_data[i][j])
                answer.append(a_data[i][j])

    problems = np.arange(1, 3163, dtype=float)
    problem2id = {p: i for i, p in enumerate(problems)}

    problem_num = list(problem2id.items())[-1][1] + 1
    count_correct = [0 for i in range(problem_num)]
    count_inter = [0 for i in range(problem_num)]
    total_correct = 0

    for i in range(len(problem)):
        pro = problem[i]
        pro_id = problem2id[pro]
        count_inter[pro_id] += 1

        if answer[i] == 1:
            count_correct[pro_id] += 1
            total_correct += 1

    normal_diff = [correct / inter if inter != 0 else 0.5 for correct, inter in zip(count_correct, count_inter)]

    # bayes diff
    # 𝑊𝐷=𝑛𝑒/(𝑛𝑒+𝑚) 𝐷+𝑚/(𝑛𝑒+𝑚) 𝑇𝐷
    TD = total_correct / len(problem)
    # TD = sum(normal_diff) / len(normal_diff)
    # avg answer time
    m = len(problem) / problem_num

    Bayes_diff = [count_inter[i] / (count_inter[i] + m) * normal_diff[i] + m / (count_inter[i] + m) * TD for i in
                  range(problem_num)]

    # problem2diff = dict()

    problem2diff = dict(zip(problems, Bayes_diff))
    return problem2diff


def compute_e_diff(problem2diff, e_data):
    [rows, cols] = e_data.shape
    problem_diff = [[0 for col in range(cols)] for row in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if e_data[i][j] == 0:
                break
            else:
                problem_diff[i][j] = problem2diff[e_data[i][j]]

    problem_diff = np.array(problem_diff)
    return problem_diff


def train_one_epoch(net, optimizer, criterion, batch_size, a_data, e_data, it_data, at_data, c_data, ca_data, recent_c):
    net.train()
    n = int(math.ceil(len(e_data) / batch_size))
    shuffled_ind = np.arange(e_data.shape[0])
    # np.random.shuffle(shuffled_ind)
    e_data = e_data[shuffled_ind]
    at_data = at_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    it_data = it_data[shuffled_ind]
    c_data = c_data[shuffled_ind]
    ca_data = ca_data[shuffled_ind]
    recent_c = recent_c[shuffled_ind]

    problem2diff = compute_diff(e_data, a_data)
    e_diff = compute_e_diff(problem2diff, e_data)
    pred_list = []
    target_list = []

    save_all_learning_gather = []

    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        c_one_seq = c_data[idx * batch_size: (idx + 1) * batch_size, :]
        ca_one_seq = ca_data[idx * batch_size: (idx + 1) * batch_size, :]
        recent_c_one_seq = recent_c[idx * batch_size: (idx + 1) * batch_size, :]
        e_diff_one_seq = e_diff[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_c = torch.from_numpy(c_one_seq).long().to(device)
        input_ca = torch.from_numpy(ca_one_seq).long().to(device)
        input_recent_c = torch.from_numpy(recent_c_one_seq).long().to(device)
        input_e_diff = torch.from_numpy(e_diff_one_seq).to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)

        pred, save_all_learning = net(input_e, input_at, target, input_it, input_c, input_ca, input_recent_c, input_e_diff)
        save_all_learning_gather.append(save_all_learning)

        mask = input_e[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    save_all_learning_gather = torch.cat(save_all_learning_gather, dim=0)

    return loss, auc, accuracy, problem2diff, save_all_learning_gather


def test_one_epoch(net, batch_size, a_data, e_data, it_data, at_data, c_data, ca_data, recent_c, problem2diff):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    e_diff = compute_e_diff(problem2diff, e_data)

    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Testing'):
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        it_one_seq = it_data[idx * batch_size: (idx + 1) * batch_size, :]
        c_one_seq = c_data[idx * batch_size: (idx + 1) * batch_size, :]
        ca_one_seq = ca_data[idx * batch_size: (idx + 1) * batch_size, :]
        recent_c_one_seq = recent_c[idx * batch_size: (idx + 1) * batch_size, :]
        e_diff_one_seq = e_diff[idx * batch_size: (idx + 1) * batch_size, :]

        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)
        input_it = torch.from_numpy(it_one_seq).long().to(device)
        input_c = torch.from_numpy(c_one_seq).long().to(device)
        input_ca = torch.from_numpy(ca_one_seq).long().to(device)
        input_recent_c = torch.from_numpy(recent_c_one_seq).long().to(device)
        input_e_diff = torch.from_numpy(e_diff_one_seq).to(device)

        target = torch.from_numpy(a_one_seq).float().to(device)

        with torch.no_grad():
            pred, save_all_learning = net(input_e, input_at, target, input_it, input_c, input_ca, input_recent_c, input_e_diff)

            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class TCKT(KTM):
    def __init__(self, n_at, n_it, n_exercise, n_concept, d_a, d_e, d_k, q_matrix, batch_size, dropout=0.2):
        super(TCKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.tckt_net = TCKTNet(n_at, n_it, n_exercise, n_concept, d_a, d_e, d_k, q_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.tckt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0
        best_save_all_learning = []
        for idx in range(epoch):
            train_loss, train_auc, train_accuracy, problem2diff, save_all_learning = train_one_epoch(self.tckt_net, optimizer, criterion,
                                                                    self.batch_size, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            logger.info("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))

            if train_auc > best_train_auc:
                best_train_auc = train_auc

            best_save_all_learning = save_all_learning

            scheduler.step()

            if test_data is not None:
                test_loss, test_auc, test_accuracy = self.eval(test_data, problem2diff)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, test_auc, test_accuracy))
                logger.info("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, test_auc, test_accuracy) + '\n')

                if test_auc > best_test_auc:
                    best_test_auc = test_auc

            torch.save(best_save_all_learning, './global_dict/all_save_all_learning.pt')

        return best_train_auc, best_test_auc, problem2diff

    def eval(self, test_data, problem2diff) -> ...:
        self.tckt_net.eval()
        return test_one_epoch(self.tckt_net, self.batch_size, *test_data, problem2diff)

    def save(self, filepath) -> ...:
        torch.save(self.tckt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.tckt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

