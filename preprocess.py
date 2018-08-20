# coding:utf-8
import os
import codecs
from hangul import split_syllables

# Datas from https://github.com/steveash/NETransliteration-COLING2018/tree/master/data
korean_datas = codecs.open('datas/wd_korean.normalized.aligned.tokens', 'r', 'utf-8').read().splitlines()
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
with codecs.open('datas/korean_decomposed.txt', 'a', 'utf-8') as fout:
    for data in korean_datas:
        flag = True
        splits = data.split('\t')
        eng_name = splits[0]
        kor_name = splits[1]
        eng_name = eng_name.replace("'", "")
        eng_name = eng_name.replace('"', '')
        kor_name = kor_name.replace("'", "")
        kor_name = kor_name.replace('"', '')
        kor_name = ''.join(kor_name.split(' '))
        kor_name_decomposed = split_syllables(unicode(kor_name))
        # remove name which contains none-roman-alphabets
        for char in eng_name:
            if char not in alphabets:
                flag = False
        # remove name which contains numbers
        for char in kor_name:
            if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                flag = False
        if flag:
            fout.write(eng_name + '\t' + kor_name_decomposed + '\n')
