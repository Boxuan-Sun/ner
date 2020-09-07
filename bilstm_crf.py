# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import bio_to_json
from bert import modeling

class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.lstm_dim = config["lstm_dim"]
        self.num_tags = config["num_tags"]
        self.is_train = config["train"]
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        self.pre_train_model = config["pre_train_model"]
        self.bert_config_json = config["bert_config_json"]

        # add placeholders for the model
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # dropout keep prob
        self.dropout = config["dropout_keep"]
        
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)  #序列的真实长度
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1] #序列最大长度

        # embeddings for chinese character and segmentation representation
        embedding = self.bert_embedding(self.is_train)
        #self.embedding = embedding
        output_layer = embedding
        
        # apply dropout before feed to lstm layer
        if self.is_train:
            output_layer = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        output = self.biLSTM_layer(output_layer, self.lstm_dim, self.lengths)
        #output = tf.layers.dense(output_layer, self.lstm_dim*2)
        # logits for tags
        self.logits = self.project_layer(output)
        
        # loss of the model
        self.loss, self.pred_ids = self.loss_layer(self.logits, self.lengths)

        # bert模型参数初始化的地方
        init_checkpoint = self.pre_train_model
        # 获取模型中所有的训练参数。
        tvars = tf.trainable_variables()
        # 加载BERT模型
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        # 打印加载模型的参数
        train_vars = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            else:
                train_vars.append(var)
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)
        if self.is_train:
            with tf.variable_scope("optimizer"):
                optimizer = self.config["optimizer"]
                if optimizer == "adam":
                    self.opt = tf.train.AdamOptimizer(self.lr)
                else:
                    raise KeyError

                grads = tf.gradients(self.loss, train_vars)
                (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

                self.train_op = self.opt.apply_gradients(
                zip(grads, train_vars), global_step=self.global_step)
            #capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            #self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )
            
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        
    def bert_embedding(self, is_train):
        # load bert embedding
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_json)  # 配置文件地址。

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_train,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        embedding = model.get_sequence_output()
        return embedding

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name="bilstm"):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)
        
    def project_layer(self, lstm_outputs, name="output"):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
            
     def loss_layer(self, project_logits, lengths, name="crf_output"):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            pred_ids, _ = tf.contrib.crf.crf_decode(potentials=logits, transition_params=self.trans, sequence_length=lengths+1)

            return tf.reduce_mean(-log_likelihood), pred_ids
            
    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, segment_ids, chars, mask, tags = batch
        feed_dict = {
            self.input_ids: np.asarray(chars),
            self.input_mask: np.asarray(mask),
            self.segment_ids: np.asarray(segment_ids),
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)

        return feed_dict
        
    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)

        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, pred_ids = sess.run([self.lengths, self.pred_ids], feed_dict)
            return lengths, pred_ids
            
     def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            labels = batch[-1]
            lengths, batch_paths = self.run_step(sess, False, batch)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [id_to_tag[int(x)] for x in labels[i][1:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][2:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results
        
     def evaluate_line(self, sess, inputs, id_to_tag):
        lengths, batch_paths = self.run_step(sess, False, inputs)
        print(batch_paths)
        tags = [id_to_tag[idx] for idx in batch_paths[0][2:lengths[0]]]
        return bio_to_json(inputs[0], tags)
