import numpy as np
import json
import argparse
import random
import time

matrix = list()
matrix_addr = list()
matrix_subj = list()
matrix_cap = list()
matrix_date = list()
N_ham = 0
N_spam = 0
P_ham = 0
P_spam = 0
vocab_size = 0
address_size = 0
subject_size = 0
capital_size = 0
html_ham = 0
html_spam = 0


def read_size(statistic_path):
    global vocab_size
    global address_size
    global subject_size
    global capital_size
    global matrix
    global matrix_addr
    global matrix_subj
    global matrix_cap
    global matrix_date
    with open(statistic_path) as f:
        lines = f.readlines()
        vocab_size = int(lines[0].strip())
        address_size = int(lines[1].strip())
        subject_size = int(lines[2].strip())
        capital_size = int(lines[3].strip())
    matrix = [[0, 0] for i in range(vocab_size)]
    matrix_addr = [[0, 0] for i in range(address_size)]
    matrix_subj = [[0, 0] for i in range(subject_size)]
    matrix_cap = [[0, 0] for i in range(capital_size)]
    matrix_date = [[0, 0] for i in range(7)]


def train(data, args):
    global matrix
    global P_ham
    global P_spam
    global N_ham
    global N_spam
    global html_ham
    global html_spam
    N_ham = 0
    for line in data:
        d = json.loads(line)
        bow = d['bow_dedup']
        label = d['label']
        receive = d['received']
        subjects = d['subject']
        capitals = d['capital']
        date = d['date']
        html = d['html']
        if label:
            N_ham += 1
        if args.use_bow:
            for w in bow:
                if label:
                    matrix[w][0] += 1
                else:
                    matrix[w][1] += 1
        if args.use_receive:
            for r in receive:
                if label:
                    matrix_addr[r][0] += 1
                else:
                    matrix_addr[r][1] += 1
        if args.use_subject:
            for sub in subjects:
                if label:
                    matrix_subj[sub][0] += 1
                else:
                    matrix_subj[sub][1] += 1
        if args.use_date:
            if label:
                matrix_date[date][0] += 1
            else:
                matrix_date[date][1] += 1
        if args.use_capital:
            for c in capitals:
                if label:
                    matrix_cap[c][0] += 1
                else:
                    matrix_cap[c][1] += 1
        if args.use_html and html:
            if label:
                html_ham += 1
            else:
                html_spam += 1

    P_ham = N_ham / len(data)
    P_spam = 1 - P_ham
    N_spam = len(data) - N_ham


def test(data, args):
    global matrix
    size = len(data)
    alpha = args.alpha
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for line in data:
        ham = np.log(P_ham)
        spam = np.log(P_spam)
        d = json.loads(line)
        bow = d['bow_dedup']
        label = d['label']
        receive = d['received']
        subjects = d['subject']
        capitals = d['capital']
        date = d['date']
        html = d['html']
        """
        Note: parameter M is set to be the size of train set to obtain better performance
        Please see report for a detailed explanation
        """
        if args.use_bow:
            for w in bow:
                ham += np.log((matrix[w][0] + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((matrix[w][1] + alpha) / (N_spam + alpha * (N_ham + N_spam)))
        if args.use_receive:
            for r in receive:
                ham += np.log((matrix_addr[r][0] + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((matrix_addr[r][1] + alpha) / (N_spam + alpha * (N_ham + N_spam)))
        if args.use_subject:
            for s in subjects:
                ham += np.log((matrix_subj[s][0] + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((matrix_subj[s][1] + alpha) / (N_spam + alpha * (N_ham + N_spam)))
        if args.use_date:
            ham += np.log((matrix_subj[date][0] + alpha) / (N_ham + alpha * (N_ham + N_spam)))
            spam += np.log((matrix_subj[date][1] + alpha) / (N_spam + alpha * (N_ham + N_spam)))

        if args.use_capital:
            for c in capitals:
                ham += np.log((matrix_cap[c][0] + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((matrix_cap[c][1] + alpha) / (N_spam + alpha * (N_ham + N_spam)))
        if args.use_html:
            if html:
                ham += np.log((html_ham + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((html_spam + alpha) / (N_spam + alpha * (N_ham + N_spam)))
            else:
                ham += np.log((N_ham - html_ham + alpha) / (N_ham + alpha * (N_ham + N_spam)))
                spam += np.log((N_spam - html_spam + alpha) / (N_spam + alpha * (N_ham + N_spam)))

        predict = 1 if ham > spam else 0
        if predict == label:
            correct += 1
            if predict == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predict == 1:
                fp += 1
            else:
                fn += 1
    accuracy = correct / size
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    random.seed(2020)  # set fixed seed to obtain same result
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)   # the same as out_path in preprocess.py
    parser.add_argument("--statistic_path", required=True)  # the same as statistic_path in preprocess.py
    parser.add_argument("--train_percentage", default=1, type=float)
    parser.add_argument("--use_bow", default=True, type=bool)
    parser.add_argument("--use_receive", default=False, type=bool)
    parser.add_argument("--use_subject", default=False, type=bool)
    parser.add_argument("--use_capital", default=False, type=bool)
    parser.add_argument("--use_html", default=False, type=bool)
    parser.add_argument("--use_date", default=False, type=bool)
    parser.add_argument("--alpha", default=1, type=float)
    args = parser.parse_args()

    read_size(args.statistic_path)
    f = open(args.data_path)
    all_data = f.readlines()
    f.close()
    random.shuffle(all_data)
    size = int(len(all_data) / 5)   # size of one fold
    a = 0
    p = 0
    r = 0
    f = 0
    start_time = time.time()
    for i in range(5):  # 5-fold cross validation
        test_set = all_data[i * size: (i + 1) * size]
        train_set = all_data[: i * size] + all_data[(i + 1) * size:]
        if args.train_percentage != 1:
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            for j in range(5):
                random.shuffle(train_set)
                sample_set = train_set[:int(args.train_percentage * len(train_set))]
                train(sample_set, args)
                aa, pp, rr, ff = test(test_set, args)
                accuracy += aa
                precision += pp
                recall += rr
                f1 += ff
                for k in range(len(matrix)):
                    matrix[k] = [0, 0]
            accuracy /= 5
            precision /= 5
            recall /= 5
            f1 /= 5
            print("accuracy: %.4f precision: %.4f recall: %.4f f1: %.4f" % (accuracy, precision, recall, f1))
        else:
            train(train_set, args)
            accuracy, precision, recall, f1 = test(test_set, args)
            print("accuracy: %.4f precision: %.4f recall: %.4f f1: %.4f" % (accuracy, precision, recall, f1))
        a += accuracy
        p += precision
        r += recall
        f += f1
        # reset
        for j in range(len(matrix)):
            matrix[j] = [0, 0]
        for j in range(len(matrix_addr)):
            matrix_addr[j] = [0, 0]
        for j in range(len(matrix_subj)):
            matrix_subj[j] = [0, 0]
        for j in range(len(matrix_cap)):
            matrix_cap[j] = [0, 0]
        for j in range(7):
            matrix_date[j] = [0, 0]
        html_spam = 0
        html_ham = 0
    a /= 5
    p /= 5
    r /= 5
    f /= 5
    print("average: \naccuracy: %.4f precision: %.4f recall: %.4f f1: %.4f" % (a, p, r, f))
    print("time elapsed: %.2f" % (time.time() - start_time))
