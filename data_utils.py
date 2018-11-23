# coding=utf-8

import random
import numpy as np

PAD_PLACEHOLDER = '<PAD>'
EOS_PLACEHOLDER = '<EOS>'
UNK_PLACEHOLDER = '<UNK>'
SOS_PLACEHOLDER = '<SOS>'
O_PLACEHOLDER = 'O'

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维

def data_reader():
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()

    return train_data, test_data

# From: 'BOS i want to fly from baltimore to dallas round trip EOS
# \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
# To: [sentence，label，intent]
def data_preprocesser(data, max_len = 50):
    # 去掉'\n'
    data = [t[:-1] for t in data]  

    split_data = []
    for line in data:
        # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注
        sentence = line.split("\t")[0].split(" ")[1:-1]
        label = line.split("\t")[1].split(" ")[1:-1]
        intent = line.split("\t")[1].split(" ")[-1]

        # print len(sentence), len(label), len(intent)

        split_data.append([sentence, label, intent])

    sentence_list, label_list, intent_list = list(zip(*split_data))

    print len(sentence_list), len(label_list), len(intent_list)

    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    padded_sentence_list = []
    padded_label_list = []
    for i in range(len(sentence_list)):
        temp_sentence = sentence_list[i]
        temp_label = label_list[i]

        if len(temp_sentence) > max_len:
            temp_sentence = temp_sentence[:max_len]
            temp_sentence[-1] = EOS_PLACEHOLDER

            temp_label = temp_label[:max_len]
            temp_label[-1] = EOS_PLACEHOLDER
        else:
            temp_sentence.append(EOS_PLACEHOLDER)
            temp_label.append(EOS_PLACEHOLDER)

            while len(temp_sentence) < max_len:
                temp_sentence.append(PAD_PLACEHOLDER)
                temp_label.append(PAD_PLACEHOLDER)

        padded_sentence_list.append(temp_sentence)
        padded_label_list.append(temp_label)

    return list(zip(padded_sentence_list, padded_label_list, intent_list))

def index_dict_generator(data):
    sentence_list, label_list, intent_list = list(zip(*data))

    sentence_vacab = set(flatten(sentence_list))
    label_vacab = set(flatten(label_list))
    intent_vacab = set(intent_list)

    word_index_dict = {PAD_PLACEHOLDER: 0, UNK_PLACEHOLDER: 1, SOS_PLACEHOLDER: 2, EOS_PLACEHOLDER: 3}
    for token in sentence_vacab:
        if token not in word_index_dict.keys():
            word_index_dict[token] = len(word_index_dict)

    index_word_dict = {v: k for k, v in word_index_dict.items()}

    label_index_dict = {PAD_PLACEHOLDER: 0, UNK_PLACEHOLDER: 1, O_PLACEHOLDER: 2}
    
    for label in label_vacab:
        if label not in label_index_dict.keys():
            label_index_dict[label] = len(label_index_dict)
        
    index_label_dict = {v: k for k, v in label_index_dict.items()}

    intent_index_dict = {UNK_PLACEHOLDER: 0}
    for intent in intent_vacab:
        if intent not in intent_index_dict.keys():
            intent_index_dict[intent] = len(intent_index_dict)
        
    index_intent_dict = {v: k for k, v in intent_index_dict.items()}

    return word_index_dict, index_word_dict, label_index_dict, index_label_dict, intent_index_dict, index_intent_dict

def to_index_convertor(data, word_index_dict, label_index_dict, intent_index_dict):
    sentence_list, label_list, intent_list = list(zip(*data))

    indexed_sentence_list = []
    indexed_label_list = []
    indexed_intent_list = []
    actual_length_list = []

    for sentence in sentence_list:
        actual_length = sentence.index(EOS_PLACEHOLDER)
        indexed_sentence = list(map(lambda word: word_index_dict[word] if word in word_index_dict else word_index_dict[UNK_PLACEHOLDER], sentence))

        actual_length_list.append(actual_length)
        indexed_sentence_list.append(indexed_sentence)
    
    for label in label_list:
        indexed_label = list(map(lambda label: label_index_dict[label] if label in label_index_dict else label_index_dict[UNK_PLACEHOLDER], label))
        indexed_label_list.append(indexed_label)

    for intent in intent_list:
        indexed_intent = intent_index_dict[intent] if intent in intent_index_dict else intent_index_dict[UNK_PLACEHOLDER]
        indexed_intent_list.append(indexed_intent)


    return list(zip(indexed_sentence_list, actual_length_list, indexed_label_list, indexed_intent_list))

def batch_data_generator(batch_size, indexed_data):
    random.shuffle(indexed_data)

    s_index = 0
    e_index = batch_size

    while e_index < len(indexed_data):
        batch_data = indexed_data[s_index:e_index]
        s_index = e_index
        e_index = e_index + batch_size

        yield batch_data
