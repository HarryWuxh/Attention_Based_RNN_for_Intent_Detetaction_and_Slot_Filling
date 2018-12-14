# coding=utf-8

import random
import numpy as np
import tensorflow as tf
import sys
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
from tensorflow.contrib import layers

class Model:
    def __init__(self, sentence_vocab_size, label_vocab_size, intent_vocab_size, input_steps=50, embedding_size=300, hidden_size=128, batch_size=16):
        # init params
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.sentence_vocab_size = sentence_vocab_size
        self.label_vocab_size = label_vocab_size
        self.intent_vocab_size = intent_vocab_size
        self.batch_size = batch_size

        print sentence_vocab_size, label_vocab_size, intent_vocab_size

        # init placeholder
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size], name='encoder_inputs') # since time_major=True, the first dim should be input_steps 
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size], name='encoder_inputs_actual_length')
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps], name='decoder_targets')
        self.intent_targets = tf.placeholder(tf.int32, [batch_size], name='intent_targets')

    def build(self):
        # embedding 
        # (sentence_vocab_size, embedding_size)
        self.embeddings = tf.Variable(tf.random_uniform([self.sentence_vocab_size, self.embedding_size], -0.1, 0.1), dtype=tf.float32, name='embeddings')

        # (input_steps, batch_size, embedding_size)
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        # encoder
        encoder_fw_cell = DropoutWrapper(LSTMCell(self.hidden_size), output_keep_prob=0.5)
        encoder_bw_cell = DropoutWrapper(LSTMCell(self.hidden_size), output_keep_prob=0.5)

        # [T*B*D], [T*B*D], [B*D], [B*D]
        # the first dim of ouput and input should be the same
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell,
                                        cell_bw=encoder_bw_cell,
                                        inputs=self.encoder_inputs_embedded,
                                        sequence_length=self.encoder_inputs_actual_length,
                                        dtype=tf.float32, time_major=True)

        encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
        encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
        encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        # decoder
        # intent logits
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_vocab_size], -0.1, 0.1), dtype=tf.float32, name='intent_W')
        intent_b = tf.Variable(tf.zeros([self.intent_vocab_size]), dtype=tf.float32, name='intent_b')

        # (batch_size, intent_vocab_size)
        intent_logits = tf.add(tf.matmul(encoder_final_state_h, intent_W), intent_b)

        # (batch_size)
        self.intent = tf.argmax(intent_logits, axis=1)

        # labelling logits
        eos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='EOS') * 2 # index of O_PLACEHOLDER is 2
        pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD') # index of PAD_PLACEHOLDER is 0
        #eos_step_embedded = tf.nn.embedding_lookup(self.embeddings, eos_time_slice)
        #pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)

        # replace outputs from previous step in initial_fn with the index of O_PLACEHOLDER
        init_step_input = tf.ones([self.batch_size, self.label_vocab_size], dtype=tf.float32) * 2 
        # padding in next_input_fn with the index of PAD_PLACEHOLDER
        pad_step_input = tf.zeros([self.batch_size, self.hidden_size * 2 + self.label_vocab_size], dtype=tf.float32) 

        # define helper for decoder
        def initial_fn():
            initial_elements_finished = (0 >= self.encoder_inputs_actual_length) # all false
            initial_input = tf.concat((init_step_input, tf.add(encoder_outputs[0], encoder_outputs[-1])), axis=1)

            return initial_elements_finished, initial_input
        
        def sample_fn(time, outputs, state):
            sample_ids = tf.to_int32(tf.argmax(outputs, axis=1))
            return sample_ids

        def next_input_fn(time, outputs, state, sample_ids):
            # pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
            # next_inputs = tf.concat((pred_embedding, encoder_outputs[time]), axis=1)

            next_inputs = tf.concat((outputs, encoder_outputs[time]), axis=1)

            element_finished = (time >= self.encoder_inputs_actual_length)
            all_finished = tf.reduce_all(element_finished)

            next_inputs = tf.cond(all_finished, lambda: pad_step_input, lambda: next_inputs)
            next_state = state

            return element_finished, next_inputs, next_state

        decoder_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_input_fn)

        memory = tf.transpose(encoder_outputs, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size, memory=memory, memory_sequence_length=self.encoder_inputs_actual_length)
        decoder_cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
        attn_wrapper = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)
        output_proj_wrapper = tf.contrib.rnn.OutputProjectionWrapper(attn_wrapper, self.label_vocab_size)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=output_proj_wrapper, helper=decoder_helper, initial_state=output_proj_wrapper.zero_state(dtype=tf.float32, batch_size=self.batch_size))
        
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=True, impute_finished=True, maximum_iterations=self.input_steps)
        self.labelling = decoder_outputs.sample_id

        # loss & optimizer
        # output_time_major=True, therefore
        # decoder_outputs.rnn_output ~ (max steps of this batch, batch_size, num_units)
        # self.mask ~ (max steps of this batch, batch_size)
        decoder_max_steps, _, _ = tf.unstack(tf.shape(decoder_outputs.rnn_output))
        decoder_targets_time_majored = tf.transpose(self.decoder_targets, [1, 0])
        self.decoder_targets_actual_length = decoder_targets_time_majored[:decoder_max_steps]
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_actual_length, 0))

        loss_slot = tf.contrib.seq2seq.sequence_loss(logits=decoder_outputs.rnn_output, targets=self.decoder_targets_actual_length, weights=self.mask)
        loss_intent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.intent_targets, depth=self.intent_vocab_size, dtype=tf.float32), logits=intent_logits))
        self.loss = loss_intent + loss_slot

        optimizer = tf.train.AdamOptimizer()
        # do Gradient Clipping
        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        clip_grads, _ = tf.clip_by_global_norm(grads, 5)
        self.train_op = optimizer.apply_gradients(zip(clip_grads, vars))

    def step(self, sess, mode, train_batch):
        unziped_batch = list(zip(*train_batch))

        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.intent, self.labelling]
            feed_dicts = {
                self.encoder_inputs: np.transpose(unziped_batch[0], [1, 0]),
                self.encoder_inputs_actual_length: unziped_batch[1],
                self.decoder_targets: unziped_batch[2],
                self.intent_targets: unziped_batch[3]
            }
        elif mode == 'test':
            output_feeds = [self.intent, self.labelling]
            feed_dicts = {
                self.encoder_inputs: np.transpose(unziped_batch[0], [1, 0]),
                self.encoder_inputs_actual_length: unziped_batch[1]
            }

        result = sess.run(output_feeds, feed_dict=feed_dicts)
        return result
        
            

