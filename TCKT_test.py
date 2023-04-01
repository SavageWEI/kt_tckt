
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0, 1, 2, ...
import time
import logging
import numpy as np
from load_data_test_recent import DATA
from TCKT_function.TCKT_test_recent import TCKT, py_name
from utils.utils import logger_fn

print("chall_test1")
py_name = py_name
logger = logger_fn("tflog", "logs/{0}/training-{1}.log".format(py_name, time.asctime()).replace(':', '_'))

def generate_q_matrix(path, n_skill, n_problem, gamma=0.0):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            problem2skill = eval(line)
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        q_matrix[p][problem2skill[p]] = 1

    return q_matrix

batch_size = 32
n_at = 1326
n_it = 2839
n_concept = 102 
n_exercise = 3162 

seqlen = 500
d_k = 128
d_a = 50
d_e = 128
q_gamma = 0.03 
dropout = 0.2

dilim = '-' * 100
logger.info("#Hyperparameter#  n_at: %d, n_it: %d, n_concept: %d, n_exercise: %d\n" % (n_at, n_it, n_concept, n_exercise) + dilim)
logger.info("#Hyperparameter#  batch_size: %d, seqlen: %d, d_k: %d, d_a: %d, d_e: %d, q_gamma: %.2f, dropout: %.1f\n" % (batch_size, seqlen, d_k, d_a, d_e, q_gamma, dropout) + dilim)


q_matrix = generate_q_matrix(
    './data/ASSISTChall/problem2skill',
    n_concept, n_exercise,
    q_gamma
)
dat = DATA(seqlen=seqlen, separate_char=',')

logging.getLogger().setLevel(logging.INFO)

# k-fold cross validation
k, train_auc_sum, valid_auc_sum = 5, .0, .0
for i in range(k):
    tckt = TCKT(n_at, n_it, n_exercise, n_concept, d_a, d_e, d_k, q_matrix, batch_size, dropout)
    train_data = dat.load_data('./data/ASSISTChall/train' + str(i) + '.txt')
    valid_data = dat.load_data('./data/ASSISTChall/valid' + str(i) + '.txt')
    best_train_auc, best_valid_auc = tckt.train(train_data, valid_data, epoch=13, lr=0.003, lr_decay_step=10)
    print('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    logger.info('fold %d, train auc %f, valid auc %f' % (i, best_train_auc, best_valid_auc))
    train_auc_sum += best_train_auc
    valid_auc_sum += best_valid_auc
print('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k))
logger.info('%d-fold validation: avg of best train auc %f, avg of best valid auc %f' % (k, train_auc_sum / k, valid_auc_sum / k))

# train and pred
train_data = dat.load_data('./data/ASSISTChall/train0.txt')
valid_data = dat.load_data('./data/ASSISTChall/valid0.txt')
test_data = dat.load_data('./data/ASSISTChall/test.txt')
tckt = TCKT(n_at, n_it, n_exercise, n_concept, d_a, d_e, d_k, q_matrix, batch_size, dropout)
tckt.train(train_data, valid_data, epoch=20, lr=0.003, lr_decay_step=10)
tckt.save("lpkt.params")

tckt.load("lpkt.params")
_, auc, accuracy = tckt.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
logger.info("#Test# auc: %.6f, accuracy: %.6f" % (auc, accuracy))