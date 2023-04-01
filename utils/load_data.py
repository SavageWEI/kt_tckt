import numpy as np
import math
import os
import torch.utils.data
import torch.nn.utils
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 0, 1, 2, ...

class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    concept sequence
    '''

    def load_data(self, path):
        f_data = open(path, 'r')
        a_data = []
        e_data = []
        it_data = []
        at_data = []
        c_data = []
        n_concept = 102

        for lineID, line in enumerate(f_data):
            line = line.strip()
            if lineID % 6 != 0:
                line_data = line.split(self.separate_char)
                if len(line_data[len(line_data) - 1]) == 0:
                    line_data = line_data[:-1]

            if lineID % 6 == 1:
                C = line_data
            elif lineID % 6 == 2:
                A = line_data
            elif lineID % 6 == 3:
                E = line_data
            elif lineID % 6 == 4:
                IT = line_data
            elif lineID % 6 == 5:
                AT = line_data

                # start split the data
                n_split = 1
                total_len = len(C)
                if total_len > self.seqlen:
                    n_split = math.floor(len(C) / self.seqlen)
                    if total_len % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    answer_sequence = []
                    exercise_sequence = []
                    it_sequence = []
                    at_sequence = []
                    concept_sequence = []

                    if k == n_split - 1:
                        end_index = total_len
                    else:
                        end_index = (k + 1) * self.seqlen
                    # choose the sequence length is larger than 2
                    if end_index - k * self.seqlen > 2:

                        for i in range(k * self.seqlen, end_index):
                            # answer_sequence.append(float(A[i]))
                            concept_sequence.append(int(C[i]))
                            answer_sequence.append(int(float(A[i])))
                            exercise_sequence.append(int(E[i]))
                            it_sequence.append(int(IT[i]))
                            at_sequence.append(int(AT[i]))

                        # print('instance:-->', len(instance),instance)
                        a_data.append(answer_sequence)
                        e_data.append(exercise_sequence)
                        it_data.append(it_sequence)
                        at_data.append(at_sequence)
                        c_data.append(concept_sequence)
        f_data.close()
        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        a_dataArray = np.zeros((len(a_data), self.seqlen))
        for j in range(len(a_data)):
            dat = a_data[j]
            a_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        it_dataArray = np.zeros((len(it_data), self.seqlen))
        for j in range(len(it_data)):
            dat = it_data[j]
            it_dataArray[j, :len(dat)] = dat

        at_dataArray = np.zeros((len(at_data), self.seqlen))
        for j in range(len(at_data)):
            dat = at_data[j]
            at_dataArray[j, :len(dat)] = dat

        c_dataArray = np.zeros((len(c_data), self.seqlen))
        for j in range(len(c_data)):
            dat = c_data[j]
            c_dataArray[j, :len(dat)] = dat

        ca_dataArray = np.zeros((len(c_data), self.seqlen))
        for i in range(0, len(c_data)):
            for j in range(0, len(c_data[1])):
                ca_dataArray[i][j] = c_dataArray[i][j] + a_dataArray[i][j] * n_concept

        # qa_dataArray = np.zeros((len(c_data), self.seqlen))
        # for i in range(0, len(c_data)):
        #     qids = np.array(list(map(int, c_data[i])))
        #     correct = np.array(list(map(int, a_data[i])))
        #     qa = qids + correct * n_concept
        #
        #     q = np.ones(self.seqlen, dtype=int) * n_concept
        #     qa2 = np.ones(self.seqlen, dtype=int) * (n_concept * 2 + 1)
        #     correct2 = np.ones(self.seqlen, dtype=int) * -1
        #     mask = np.zeros(self.seqlen, dtype=int)
        #
        #     q[: len(qids)] = qids
        #     qa2[: len(qa)] = qa
        #     correct2[: len(correct)] = correct
        #     mask[: len(qa)] = np.ones(len(qa), dtype=int)
        #
        #     a = torch.cat(
        #         (torch.LongTensor([2 * n_concept]), torch.LongTensor(qa2[:-1]))
        #     ),
        #
        #     dat = a
        #     c_dataArray[j, :len(dat)] = dat


        return a_dataArray, e_dataArray, it_dataArray, at_dataArray, c_dataArray, ca_dataArray




