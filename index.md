---
layout: single
toc: true
---

This site accompanies the latter half of the [ART.T458: Machine Learning](http://www.ocw.titech.ac.jp/index.php?module=General&action=T0300&JWC=201804845&lang=EN) course held in Tokyo Institute of Technology, which focuses on Deep Learning with flavor of Natural Language Processing. We distribute the course materials, demos, and implementation examples related to the course.

## Lecture 1: Feedforward Neural Network (I)

+ [Slides](https://speakerdeck.com/chokkan/feedforward-neural-network-i-binary-classification)
+ Demo: [Interactive single-layer perceptron](demo-slp.html)
+ Demo: [Interactive multi-layer perceptron](demo-mlp.html)
+ Implementation: on [github](https://github.com/chokkan/deeplearning/blob/master/notebook/binary.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/binary.ipynb)
+ Keywords: binary classification, Threshold Logic Units (TLUs), Single-layer Perceptron (SLP), Perceptron algorithm, sigmoid function, Stochastic Gradient Descent (SGD), Multi-layer Perceptron (MLP), Backpropagation, Computation Graph, Automatic Differentiation, Universal Approximation Theorem

## Lecture 2: Feedforward Neural Network (II)

+ [Slides](https://speakerdeck.com/chokkan/feedforward-neural-network-ii-multi-class-classification)
+ Implementation: on [github](https://github.com/chokkan/deeplearning/blob/master/notebook/mnist.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/mnist.ipynb)
+ Keywords: multi-class classification, linear multi-class classifier, softmax function, Stochastic Gradient Descent (SGD), mini-batch training, loss functions, activation functions, dropout

## Lecture 3: Word embeddings

+ [Slides](https://speakerdeck.com/chokkan/word-embeddings)
+ Demo: [Word vectors trained on Japanese Wikipedia](https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_ja.ipynb)
+ Demo: [Word vectors pre-trained on English newspapers](https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_en.ipynb)
+ Keywords: word embeddings, distributed representation, distributional hypothesis, pointwise mutual information, singular value decomposition, word2vec, word analogy, GloVe, fastText

## Lecture 4: DNN for structural data

+ [Slides](https://speakerdeck.com/chokkan/dnn-for-structural-data)
+ Implementation: on [Github](https://github.com/chokkan/deeplearning/blob/master/notebook/name.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/name.ipynb)
+ Keywords: Recurrent Neural Networks (RNNs), Gradient vanishing and exploding, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), Recursive Neural Network, Tree-structured LSTM, Convolutional Neural Networks (CNNs)

## Lecture 5: Encoder-decoder models

+ [Slides](https://speakerdeck.com/chokkan/encoder-decoder-models)
+ Keywords: language modeling, Recurrent Neural Network Language Model (RNNLM), encoder-decoder models, sequence-to-sequence models, attention mechanism, reading comprehension, question answering, headline generation, multi-task learning, character-based RNN, byte-pair encoding, Convolutional Sequence to Sequence (ConvS2S), Transformer, coverage
