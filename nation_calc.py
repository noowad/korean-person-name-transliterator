# coding:utf-8
from __future__ import print_function
import os
import argparse
import codecs


def calc_result(write_flag):
    files = [f for f in os.listdir('results/k4') if not f.startswith('.')]
    if 'UNK.txt' in files:
        files.remove('UNK.txt')
    all_correct_count = 0
    all_count = 0
    all_mrr = 0
    nation_count = 0
    nation_over_85_count = 0
    for f in files:
        count = 0
        correct_count = 0
        nation_count += 1
        source_names = []
        target_names = []
        candidates = []
        lines = []
        for k in ['0','1','2','3','4']:
            if os.path.exists('./results/k' + k + '/' + f):
                lines += codecs.open('results/k' + k + '/' + f, 'r', 'utf-8').read().splitlines()
        for line in lines:
            count += 1
            all_count += 1
            names = line.split('\t')
            source_names.append(names[0])
            target_names.append(names[1])
            candidates.append(names[2:][:5])
        rr_list = []
        for source_name, target_name, candidate in zip(source_names, target_names, candidates):
            rr = 0.
            for i, cand in enumerate(candidate):
                if target_name == cand:
                    correct_count += 1
                    all_correct_count += 1
                    rr = 1. / (i + 1)
                    break
            rr_list.append(rr)
            if write_flag:
                if target_name not in candidate:
                    with open('./falses/nations/' + f[:3] + '.falses', 'a') as f_false:
                        f_false.write(source_name + "\t" + target_name)
                        for num in range(0, len(candidate)):
                            f_false.write("\t" + candidate[num])
                        f_false.write("\n")
        acc = float(correct_count) / count
        mrr = sum(rr_list) / len(rr_list)
        all_mrr += mrr
        if acc >= 0.85:
            nation_over_85_count += 1
        with open('results/matome.result', 'a') as fout:
            fout.write(f[:3] + "\t" + str(correct_count) + "\t" + str(count) + "\t" + str(acc) + "\t" + str(mrr) + "\n")
    all_acc = float(all_correct_count) / all_count
    all_mrr = str(all_mrr / nation_count)
    with open('results/matome.result', 'a') as fout:
        fout.write(
            "ALL" + "\t" + str(all_correct_count) + "\t" + str(all_count) + "\t" + str(all_acc) + "\t" + all_mrr + "\n")
        fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))


def data_over_50(mode):
    count = 0
    correct_count = 0
    nation_count = 0
    nation_over_85_count = 0
    with open('./results/'+mode+'_over_50.result', 'w') as fout:
        for line in codecs.open('./results/'+mode+'_matome.result', 'r', 'utf-8').read().splitlines()[:-2]:
            nation = line.split('\t')[0]
            correct_num = int(line.split('\t')[1])
            data_num = int(line.split('\t')[2])
            result = float(line.split('\t')[3])
            if data_num >= 50:
                nation_count += 1
                count += data_num
                correct_count += correct_num
                if result >= 0.85:
                    nation_over_85_count += 1
                fout.write(line+'\n')
    acc = str(float(correct_count) / count)

    with open('./results/'+mode+'_over_50.result', 'a') as fout:
        fout.write("ALL\t" + str(correct_count) + "\t" + str(count) + "\t" + acc + "\n")
        fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))


def data_under_50(mode):
    count = 0
    correct_count = 0
    nation_count = 0
    nation_over_85_count = 0
    with open('./results/'+mode+'_under_50.result', 'w') as fout:
        for line in codecs.open('./results/'+mode+'_matome.result', 'r', 'utf-8').read().splitlines()[:-2]:
            nation = line.split('\t')[0]
            correct_num = int(line.split('\t')[1])
            data_num = int(line.split('\t')[2])
            result = float(line.split('\t')[3].strip())
            if data_num < 50:
                nation_count += 1
                count += data_num
                correct_count += correct_num
                if result >= 0.85:
                    nation_over_85_count += 1
                    fout.write(line + '\n')
    acc = str(float(correct_count) / count)

    with open('./results/'+mode+'_under_50.result', 'a') as fout:
        fout.write("ALL\t" + str(correct_count) + "\t" + str(count) + "\t" + acc + "\n")
        fout.write("Nations with Accuracy over 85%" + "\t" + str(nation_over_85_count) + "/" + str(nation_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', action='store', dest='mode', type=str, default='extra',
                        help='Enter re-train mode')
    par_args = parser.parse_args()
    mode = par_args.mode
    calc_result(write_flag=0)
    #data_over_50(mode=mode)
    #data_under_50(mode=mode)
    print("Done")
