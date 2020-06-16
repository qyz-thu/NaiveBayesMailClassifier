import json
import re
import argparse


def preprocess(label_path, out_path):
    num = 0
    has_receive = 0
    has_date = 0
    has_subject = 0
    has_from = 0
    has_body = 0
    f_w = open(out_path, 'w')
    with open(label_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = line.strip().split(' ')
            label = 1 if tokens[0] == 'ham' else 0
            data_path = tokens[1]
            data = {'label': label, "received": [], 'subject': ""}
            try:
                with open("./" + data_path[3:], 'r', encoding='utf-8') as f_d:
                    lines = f_d.readlines()
                    while len(lines) != 0:
                        if lines[0].find("Received:") >= 0:
                            received = lines.pop(0)
                            while len(lines) != 0 and re.match('\s', lines[0]) is not None:
                                received += lines.pop(0)
                            data["received"].append(received)
                        elif lines[0].find("Subject:") >= 0:
                            data["subject"] = lines.pop(0)
                        elif lines[0].find("Date:") >= 0:
                            data["date"] = lines.pop(0)
                        elif lines[0].find("From:") >= 0:
                            data["from"] = lines.pop(0)
                        elif lines[0] == '\n':
                            lines.pop(0)
                            data['body'] = ''.join(lines)
                            break
                        else:
                            lines.pop(0)
                    num += 1
            except UnicodeDecodeError:
                print(data_path + " not encoded in utf-8!")
            if len(data["received"]) != 0:
                has_receive += 1
            if "subject" in data:
                has_subject += 1
            if "date" in data:
                has_date += 1
            if "from" in data:
                has_from += 1
            if "body" in data:
                has_body += 1
            else:
                continue
            data = json.dumps(data)
            f_w.write(data)
            f_w.write('\n')
    f_w.close()
    print("successfully processed: %d mails" % num)
    print("has receive: %d" % has_receive)
    print("has subject: %d" % has_subject)
    print("has date: %d" % has_date)
    print("has from: %d" % has_from)
    print("has body: %d" % has_body)


def get_feature(data_path, out_path, statistic_path):
    vocab = dict()
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)
            pattern = re.compile("[a-zA-Z]+")
            words = pattern.findall(data['body'])
            for word in words:
                if word not in vocab:
                    vocab[word] = 0
                else:
                    vocab[word] += 1
    s_vocab = dict()
    for word in vocab:
        if vocab[word] >= 3:
            s_vocab[word] = len(s_vocab)
    # f = open(vocab_path)
    # v = f.read().strip().split('\t')
    # f.close()
    # s_vocab = dict()
    # for w in v:
    #     s_vocab[w] = len(s_vocab)
    # with open(vocab_path, 'w') as f:
    #     for w in s_vocab:
    #         f.write(w + '\t')
    print("total %d words" % len(s_vocab))
    addresses = {"<UNK>": 0}
    subject_words = {"<none>": 0}
    capitals = dict()
    with open(out_path, 'w') as f_w:
        with open(data_path) as f:
            for line in f:
                data = json.loads(line)
                pattern = re.compile("[a-zA-Z]+")
                # process bow
                words = pattern.findall(data['body'])
                word_list = [s_vocab[word] for word in words if word in s_vocab]
                data['bow'] = word_list
                word_list = list(set(word_list))
                data["bow_dedup"] = word_list
                # process receive
                receive = []
                for r in data["received"]:
                    # get domain name or "localhost" or "unknown"
                    address = re.search("from \S*([a-zA-Z0-9]+\.)+([a-zA-Z]+)", r)
                    if address is None:
                        address = re.search("(from) (unknown)", r)
                    if address is None:
                        address = re.search("(from) (localhost)", r)
                    if address is None:
                        receive.append(0)
                    else:
                        name = address.group(2).lower()
                        if name not in addresses:
                            addresses[name] = len(addresses)
                        receive.append(addresses[name])
                data["received"] = receive
                # process subject
                words = pattern.findall(data['subject'])
                for w in words:
                    if w not in s_vocab:
                        continue
                    if w not in subject_words:
                        subject_words[w] = len(subject_words)
                word_list = [subject_words[w] for w in words if w in subject_words]
                if len(word_list) == 0:
                    word_list.append(0)
                data['subject'] = word_list
                # process date
                if 'date' not in data:
                    data['date'] = 0
                else:
                    x = -1
                    time = data['date'].lower()
                    if time.find("mon") >=0 or time.find("tue") >=0 or \
                        time.find("wed") >=0 or time.find("thu") >=0 or time.find("fri") >=0:
                        x = 0
                    elif time.find("sat") >= 0 or time.find("sun") >= 0:
                        x = 1
                    else:
                        data['date'] = 0
                    if x != -1:
                        y = -1
                        pattern = re.compile("([0-9]{2}):[0-9]{2}:[0-9]{2}")
                        a = pattern.findall(time)
                        if len(a) == 0:
                            data['date'] = 0
                        else:
                            hour = int(a[0])
                            if 0 <= hour < 8:
                                y = 1
                            elif 8 <= hour < 16:
                                y = 2
                            elif 16 <= hour < 24:
                                y = 3
                        if y != -1:
                            data['date'] = 3 * x + y

                # process capital
                pattern = re.compile("[^a-zA-Z]([A-Z][A-Z]+)[^a-zA-Z]")
                caps = pattern.findall(data["body"])
                for c in caps:
                    if c not in capitals:
                        capitals[c] = len(capitals)
                cap_list = [capitals[c] for c in caps if c in capitals]
                data['capital'] = cap_list
                # process html
                pattern = re.compile("</?[A-Za-z]+>")
                html = pattern.findall(data['body'])
                data['html'] = 1 if len(html) > 6 else 0

                data.pop("body")
                data = json.dumps(data)
                f_w.write(data + '\n')
    with open(statistic_path, 'w') as f:
        f.write(str(len(s_vocab)) + '\n')
        f.write(str(len(addresses)) + '\n')
        f.write(str(len(subject_words)) + '\n')
        f.write(str(len(capitals)) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    Note: please put this file in the same dir as data & label
    or you may have to 
    """
    parser.add_argument("--label_path", required=True, help="path to label")
    parser.add_argument("--out_path", required=True, help="path of output file")
    parser.add_argument("--statistic_path", required=True, help="path to save statistics")
    args = parser.parse_args()
    preprocess(args.label_path, "./mydata")
    get_feature("./mydata", args.out_path, args.statistic_path)
