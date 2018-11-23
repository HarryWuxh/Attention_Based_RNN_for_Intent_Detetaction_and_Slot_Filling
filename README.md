# Attention_Based_RNN_for_Intent_Detetaction_and_Slot_Filling
This is a simplified tensorflow implementation of paper *Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling*.  
This paper proposes a encoder-decoder model to do the intent detection and slot filling at the same time. In this implementation, encoder is implemented with tf.nn.bidirectional_dynamic_rnn and decoder with attention is implemented with tf.contrib.seq2seq.  
If there is any question, please contact me via email or issues, harry_wxh@163.com.  
![Proposed model](https://github.com/HarryWuxh/Attention_Based_RNN_for_Intent_Detetaction_and_Slot_Filling/blob/master/img/encoder-decoder%20model.PNG)
