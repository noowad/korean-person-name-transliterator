import codecs


def levenshtein_distance(sentence1, sentence2):
    '''
    levenshtein distance between sentence1, sentence2
    :param sentence1: sentence string
    :param sentence2: sentence string
    '''
    sentence1, sentence2 = list(sentence1), list(sentence2)

    if len(sentence1) > len(sentence2):
        sentence1, sentence2 = sentence2, sentence1
    # dynamic programming
    distances = range(len(sentence1) + 1)
    for num2, word2 in enumerate(sentence2):
        distances_ = [num2 + 1]
        for num1, word1 in enumerate(sentence1):
            if word1 == word2:
                distances_.append(distances[num1])
            else:
                distances_.append(1 + min((distances[num1], distances[num1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


results = codecs.open('results/result.txt', 'r', 'utf-8').read().splitlines()
with codecs.open('datas/korean_train2.txt', 'w', 'utf-8') as fout:
    for result in results:
        splits = result.split('\t')
        eng_name = splits[0]
        correct = splits[1]
        answer = splits[2]
        dist = levenshtein_distance(correct, answer)
        if abs(len(correct) - dist) > 2:
            fout.write(eng_name + '\t' + correct + '\n')
