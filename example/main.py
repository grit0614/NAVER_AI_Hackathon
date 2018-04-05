# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

import time
import json

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        # 첫 번째 레이어
        self.fc1 = nn.Linear(self.max_length * self.embedding_dim, 200)
        # 두 번째 (아웃풋) 레이어
        self.fc2 = nn.Linear(200, 1)

    def forward(self, data: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        hidden = self.fc1(embeds.view(batch_size, -1))
        # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
        output = torch.sigmoid(self.fc2(hidden)) * 9 + 1
        return output


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=8)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    model = Regression(config.embedding, config.strmaxlen)
    if GPU_NUM:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':

        ### Hyperparameters
        VERBOSE = False
        EMBEDDING_SIZE = 512 # Word Embedding Dimension
        HIDDEN_SIZE = 128 # The number of Hidden Units
        BATCH_SIZE = 512 # should be less than #train_data
        KEEP_PROB= 0.5 # Dropout Probability
        epsilon = 5.0  # IMDB ideal norm length ???
        TransferLearningFlag = False # Use NaverSentimentMovieCorpus Data
        DataPath = "./nsmc/raw/"
        VocabMinFreq = 0
        LearningRate = 1e-3
        MaxStep = 1000
        #

        if VERBOSE:
            from tqdm import tqdm

        ### Preparing Data
        x_train = []; y_train = []
        #x_test = []; y_test = []

        ### Modify ###
        # f_traindata = open("./train_data", 'r')
        # f_trainlabel = open("./train_label", 'r')
        f_traindata = os.path.join(DATASET_PATH, 'train', 'train_data')
        f_trainlabel = os.path.join(DATASET_PATH, 'train', 'train_label')
        #f_testdata = open("./test_data", 'r')
        #f_testlabel = open("./test_label", 'r')
        #####

        def tokenizer(raw_sentence):
            
            ### Add Preprocessing
            # Remove special character
            # Add space between character if you want to build character based model
            # Add POS Tagger
            
            token = raw_sentence
            return token
        
        f_traindata = open(f_traindata, 'rt', encoding='utf-8')
        f_trainlabel = open(f_trainlabel, 'r')
        for line in f_traindata:
            x_train.append(line.strip().encode('utf-8'))
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        for line in f_trainlabel:
            y_train.append(line.strip())
                
        f_traindata.close(); f_trainlabel.close(); 

        x_train = np.array(x_train); y_train = np.array(y_train, dtype=np.int)
        # print(x_train[1])
        # print(x_train[2])
        # print(y_train[1])
        # print(y_train[2])

        # print([len(sent.split(b' ')) for sent in x_train])

        CLASS_NUM = y_train.max()
        MAX_LENGTH = np.max([len(sent.split(b' ')) for sent in x_train])
        # MAX_LENGTH = '1000'.encode()
        ### Shuffle
        train = np.hstack((x_train.reshape(-1,1), y_train.reshape(-1,1))); # np.random.shuffle(train)
        #test = np.hstack((x_test.reshape(-1,1), y_test.reshape(-1,1))); # np.random.shuffle(test)

        ### Build Vocabulary
        # VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        def TokBySpace(iterator):
            return (x.split(b' ') for x in iterator)

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_LENGTH, min_frequency=VocabMinFreq, tokenizer_fn=TokBySpace)

        # print(train[:,0])
        x_transform_train = vocab_processor.fit_transform(train[:,0])

        #x_transform_test = vocab_processor.transform(test[:,0])
        vocab = vocab_processor.vocabulary_
        vocab_size = len(vocab)
        x_train = np.array(list(x_transform_train)); y_train = np.array(train[:,1], dtype=np.int)
        #x_test = np.array(list(x_transform_test)); y_test = np.array(test[:,1], dtype=np.int)
        train_size = x_train.shape[0]
        #test_size = x_test.shape[0]

        print("Vocab Size: ", vocab_size)
        #dev_size = test_size
        #test_size -= dev_size
        print("Train size: ", train_size)
        #print("Dev size : ", dev_size)
        print("#Class : ", CLASS_NUM)
        print("MaxSentLen: ", MAX_LENGTH)

        print(x_train)
        print(y_train)
        #print(x_test)
        #print(y_test)

        def get_freq(vocabulary):
            vocab_freq = vocabulary._freq
            words = vocab_freq.keys()
            freq = [0] * vocab_size
            for word in words:
                word_idx = vocab.get(word)
                word_freq = vocab_freq[word]
                freq[word_idx] = word_freq

            return freq

        def _scale_l2(x, norm_length):
            # shape(x) = (batch, num_timesteps, d)
            # Divide x by max(abs(x)) for a numerically stable L2 norm.
            # 2norm(x) = a * 2norm(x/a)
            # Scale over the full sequence, dims (1, 2)
            alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
            l2_norm = alpha * tf.sqrt(
                tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
            x_unit = x / l2_norm
            return norm_length * x_unit

        def add_perturbation(embedded, loss):
            """Adds gradient to embedding and recomputes classification loss."""
            grad, = tf.gradients(
                loss,
                embedded,
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            grad = tf.stop_gradient(grad)
            perturb = _scale_l2(grad, epsilon)
            return embedded + perturb

        def normalize(emb, weights):
            # weights = vocab_freqs / tf.reduce_sum(vocab_freqs) ?? 这个实现没问题吗
            print("Weights: ", weights)
            mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
            var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
            stddev = tf.sqrt(1e-6 + var)
            return (emb - mean) / stddev

        graph = tf.Graph()
        with graph.as_default():
            batch_x = tf.placeholder(tf.int32, [None, MAX_LENGTH])
            batch_y = tf.placeholder(tf.float32, [None, CLASS_NUM])
            keep_prob = tf.placeholder(tf.float32)
            vocab_freqs = tf.constant(get_freq(vocab), dtype=tf.float32, shape=(vocab_size, 1))
            
            weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
            
            embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
            W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
            W_fc = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, CLASS_NUM], stddev=0.1))
            b_fc = tf.Variable(tf.constant(0., shape=[CLASS_NUM]))

            embedding_norm = normalize(embeddings_var, weights)
            batch_embedded = tf.nn.embedding_lookup(embedding_norm, batch_x)


            def cal_loss_logit(batch_embedded, keep_prob, reuse=True, scope="loss"):
                with tf.variable_scope(scope, reuse=reuse) as scope:
                    rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                                            inputs=batch_embedded, dtype=tf.float32)

                # Attention
                    H = tf.add(rnn_outputs[0], rnn_outputs[1]) # fw + bw
                    M = tf.tanh(H) # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
                    print(M.shape)
                # alpha (bs * sl, 1)
                    alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
                    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_LENGTH, 1])) # supposed to be (batch_size * HIDDEN_SIZE, 1)
                    print(r.shape)
                    r = tf.squeeze(r)
                    h_star = tf.tanh(r) # (batch , HIDDEN_SIZE
                # attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
                    drop = tf.nn.dropout(h_star, keep_prob)
                
                # Fully connected layer（dense layer)
                    y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc)
            
                return y_hat, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
            
            lr = LearningRate
            logits, cl_loss = cal_loss_logit(batch_embedded, keep_prob, reuse=False)
            embedding_perturbated = add_perturbation(batch_embedded, cl_loss)
            ad_logits, ad_loss = cal_loss_logit(embedding_perturbated, keep_prob, reuse=True)
            loss = cl_loss + ad_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

            # Accuracy metric
            prediction = tf.argmax(tf.nn.softmax(logits), 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))


        steps = MaxStep+1 # about 5 epoch

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            print("Initialized! ")

            train_labels = tf.one_hot(y_train, CLASS_NUM, 1, 0)
        #    test_labels = tf.one_hot(y_test, CLASS_NUM, 1, 0)

            y_train = train_labels.eval()
        #    y_test = test_labels.eval()

        #    dev_x = x_test[: dev_size, :]
        #    dev_y = y_test[: dev_size, :]

        #    test_x = x_test[dev_size:, :]
        #    test_y = y_test[dev_size:, :]

            offset = 0
            print("Start trainning")
            start = time.time()
            time_consumed = 0
            
            if VERBOSE:
                pbar = tqdm(total=MaxStep)
                
            for step in range(steps):
                step_start = time.time()

                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_data = x_train[offset: offset + BATCH_SIZE, :]
                batch_label = y_train[offset: offset + BATCH_SIZE, :]

                fd = {batch_x: batch_data, batch_y: batch_label, keep_prob: KEEP_PROB}
                l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)
                time_consumed += time.time() - step_start
                if step % 100 == 0:
                    print()
                    print("Minibatch average time: ", time_consumed/ 100, " s")
                    time_consumed = 0
                    print("Step %d: loss : %f   accuracy: %f %%" % (step, l, acc * 100))

                if step % 500 == 0:
                    print("******************************\n")
        #            dev_loss, dev_acc = sess.run([loss, accuracy], feed_dict={batch_x: dev_x, batch_y: dev_y, keep_prob: 1})
        #            print("Dev set at Step %d: loss : %f   accuracy: %f %%\n" % (step, dev_loss, dev_acc * 100))
                    print("******************************")
                if VERBOSE:
                    pbar.update(1)
            if VERBOSE:
                pbar.close()    
            print("Training finished, time consumed : ", time.time() - start, " s")
            # print("start predicting:  \n")
            # test_accuracy = sess.run([accuracy], feed_dict={batch_x: test_x, batch_y: test_y, keep_prob: 1})
            # print("Test accuracy : %f %%" % (test_accuracy[0]*100))


        # 데이터를 로드합니다.
        # dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        # train_loader = DataLoader(dataset=dataset,
        #                           batch_size=config.batch,
        #                           shuffle=True,
        #                           collate_fn=collate_fn,
        #                           num_workers=2)
        # MAX_LENGTH = len(train_loader)
        # epoch마다 학습을 수행합니다.
        # for epoch in range(config.epochs):
        #     avg_loss = 0.0
        #     for i, (data, labels) in enumerate(train_loader):


            #     predictions = model(data)
            #     label_vars = Variable(torch.from_numpy(labels))
            #     if GPU_NUM:
            #         label_vars = label_vars.cuda()
            #     loss = criterion(predictions, label_vars)
            #     if GPU_NUM:
            #         loss = loss.cuda()

            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     print('Batch : ', i + 1, '/', total_batch,
            #           ', MSE in this minibatch: ', loss.data[0])
            #     avg_loss += loss.data[0]
            # print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
            



        # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
        #
        nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                    train__loss=float(avg_loss/total_batch), step=epoch)
        # DONOTCHANGE (You can decide how often you want to save the model)
        nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)