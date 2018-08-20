from hyperparams import Hyperparams as hp
import codecs
import argparse
import os

def eval_result(mode, cross_k):
    count = 0
    source_names = []
    target_names = []
    candidates = []

    for line in codecs.open('./results/k'+str(cross_k)+'/'+'TUR.txt', 'r', 'utf-8').read().splitlines():
        count += 1
        source_names.append(line.split('\t')[0])
        target_names.append(line.split('\t')[1])
        candidates.append(line.split('\t')[2:][:hp.candidate_size])
    if not os.path.exists('./baseline_falses/k' + str(cross_k)):
        os.makedirs('./baseline_falses/k' + str(cross_k))
    with open('./baseline_falses/k' + str(cross_k) + '/POL.txt', 'w') as fout:
        for k in range(1, hp.candidate_size+1):
            correct_count = 0
            for source_name, target_name, candidate in zip(source_names, target_names, candidates):
                if target_name in candidate[:k]:
                    correct_count += 1
                else:
                    if k == 5:
                        fout.write(source_name + "\t" + target_name)
                        for num in range(0, len(candidate)):
                            fout.write("\t" + candidate[num])
                        fout.write("\n")
            print("top{} Accuracy:{}/{}={}".format(k, correct_count, count, float(correct_count) / count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', action='store', dest='k', type=int, default=0,
                        help='Enter cross-validation k')
    parser.add_argument('-mode', action='store', dest='mode', type=str, default='extra',
                        help='Enter re-train mode')
    par_args = parser.parse_args()
    mode = par_args.mode
    cross_k = par_args.k
    eval_result(mode, cross_k)
    print("Done")
