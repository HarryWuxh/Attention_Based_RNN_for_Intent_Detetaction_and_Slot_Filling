# coding=utf-8

from data_utils import *
from model import *

# convert indexed sentence to words
def index_to_word(indexed_data, index_dict):
    sentence = list(map(lambda index: index_dict[index] if index in index_dict else '???', indexed_data))

    return sentence

input_steps = 50
epoch_num = 50
batch_size = 16

train_data, test_data = data_reader()
preprocessed_train_data = data_preprocesser(train_data)
preprocessed_test_data = data_preprocesser(test_data)

print preprocessed_train_data[0]

word_index_dict, index_word_dict, label_index_dict, index_label_dict, intent_index_dict, index_intent_dict = index_dict_generator(preprocessed_train_data)

indexed_train_data = to_index_convertor(preprocessed_train_data, word_index_dict, label_index_dict, intent_index_dict)
indexed_test_data = to_index_convertor(preprocessed_test_data, word_index_dict, label_index_dict, intent_index_dict)
print indexed_train_data[0]
print index_intent_dict[21]

model = Model(sentence_vocab_size=len(word_index_dict), label_vocab_size=len(label_index_dict), intent_vocab_size=len(intent_index_dict), batch_size=batch_size, input_steps=input_steps)
model.build()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_num):
    # training
    train_loss = 0.0

    for i, batch_data in enumerate(batch_data_generator(batch_size, indexed_train_data)):
        _, loss, intent, labelling = model.step(sess, 'train', batch_data)

        train_loss += loss

    print "[Epoch {}] Average train loss: {}".format(epoch, train_loss / (i + 1))

    # testing for every epoch
    intent_correct_count = 0
    labelling_correct_count = 0
    labelling_total_count = 0
    for j, batch_data in enumerate(batch_data_generator(batch_size, indexed_test_data)):
        intent, labelling = model.step(sess, 'test', batch_data)

        for k in range(len(intent)):
            if intent[k] == batch_data[k][3]:
                intent_correct_count += 1

        for k in range(np.shape(labelling)[1]):
            for l in range(np.shape(labelling)[0]):
                if batch_data[k][2][l] > 2:
                    labelling_total_count += 1

                    if labelling.transpose(1, 0)[k][l] == batch_data[k][2][l]:
                        labelling_correct_count += 1                    

        # sample the fisrt testing data
        if j == 0:
            actual_length = batch_data[j][1]
            print "Input Sentence       : ", index_to_word(batch_data[j][0], index_word_dict)[:actual_length]
            print "Labelling Truth      : ", index_to_word(batch_data[j][2], index_label_dict)[:actual_length]
            print "Labelling Prediction : ", index_to_word(labelling.transpose(1, 0)[j], index_label_dict)[:actual_length]
            print "Intent Truth         : ", index_intent_dict[batch_data[j][3]]
            print "Intent Prediction    : ", index_intent_dict[intent[j]]

    print "[Epoch {}] Intent accuracy: {}".format(epoch, float(intent_correct_count) / len(indexed_test_data))
    print "[Epoch {}] Labelling accuracy: {}".format(epoch, float(labelling_correct_count) / labelling_total_count)
