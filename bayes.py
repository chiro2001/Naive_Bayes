import os
import re
import numpy as np
from functools import reduce
import random


def split_text(text: str):
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    tokens = re.split(r'\W+', text)
    # 除了单个字母，例如大写的I，其它单词变成小写
    return [tok.lower() for tok in tokens if len(tok) > 2 and not tok.isdigit()]


def get_words_data():
    class_types = [r for r in os.listdir('email') if os.path.isdir(os.path.join('email', r))]

    def read_data(filename: str) -> str:
        with open(filename, 'r', encoding='gbk') as f:
            return f.read()

    words_data = []
    for c in class_types:
        for filename in os.listdir(os.path.join('email', c)):
            file_data = read_data(os.path.join(f'email/{c}', filename))
            words_data.append((c, split_text(file_data)))
    return words_data, class_types


def get_words_label(words_data: list) -> list:
    words_label = set({})
    for words in words_data:
        words_label.update(words[1])
    res = list(words_label)
    res.sort()
    return res


def get_words_map(words_label) -> dict:
    return {val: index for index, val in enumerate(words_label)}


def get_words_vector(words: str, words_label: list) -> list:
    return [(1 if val == words else 0) if not isinstance(words, list)
            else (1 if val in words else 0) for val in words_label]


def native_bayes_train(words_label: list, train_data: list, class_types: list, alpha: float = 1e-3):
    p_result = {c: np.array([alpha for _ in range(len(words_label))]) for c in class_types}
    p_B = 0
    for data in train_data:
        words_vector = np.array(get_words_vector(data[1], words_label))
        p_result[data[0]] += words_vector + alpha
        p_B += (1 if data[0] == class_types[0] else 0) / len(train_data)
    for k in p_result:
        p_result[k] = np.log(p_result[k])
    return p_result, p_B


def native_bayes_test(words_label: list, p_result: dict, test_data: list, class_types: list, p_B: float,
                      alpha: float = 1e-3) -> float:
    accurate = 0
    for data in test_data:
        words_vector = np.array(get_words_vector(data[1], words_label))
        Qs = {key: np.log(p_B) + sum(p_result[key] * words_vector) - np.log(sum(words_vector + alpha)) for key in
              p_result}
        classification = reduce((lambda x, y: x if x[1] > y[1] else y), ((k, Qs[k]) for k in Qs))[0]
        print(f"result {classification == data[0]}, classification = {classification}, label = {data[0]}")
        accurate += (1 if classification == data[0] else 0) / len(test_data)
    return accurate



def main():
    words_data, class_types = get_words_data()
    random.shuffle(words_data)
    words_label = get_words_label(words_data)
    p_result_, p_B_ = native_bayes_train(words_label, words_data[:40], class_types)
    native_bayes_test(words_label, p_result_, words_data[40:], class_types, p_B_)


if __name__ == '__main__':
    main()
