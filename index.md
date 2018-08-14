---
layout: splash
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  cta_label: "Course (OCW)"
  cta_url: "http://www.ocw.titech.ac.jp/index.php?module=General&action=T0300&JWC=201804845&lang=EN"
excerpt: 'with flavor of Natural Language Processing (NLP)'
intro: 
  - excerpt: 'This site accompanies the latter half of the [ART.T458: Machine Learning](http://www.ocw.titech.ac.jp/index.php?module=General&action=T0300&JWC=201804845&lang=EN) course at [Tokyo Institute of Technology](https://www.titech.ac.jp/english/), which focuses on Deep Learning for Natural Language Processing (NLP). Course materials, demos, and implementations are available.'
lecture1:
  - url: https://speakerdeck.com/chokkan/feedforward-neural-network-i-binary-classification
    image_path: /assets/images/ffnn-binary.png
    alt: "Slides for lecture #1"
    title: "Open on Speaker Deck"
  - url: demo-slp.html
    image_path: /assets/images/demo-slp.png
    alt: "Interactive single-layer perceptron"
    title: "Open the demo"
  - url: demo-mlp.html
    image_path: /assets/images/demo-mlp.png
    alt: "Interactive multi-layer perceptron"
    title: "Open the demo"
lecture2:
  - url: https://speakerdeck.com/chokkan/feedforward-neural-network-ii-multi-class-classification
    image_path: /assets/images/ffnn-multi1.png
    alt: "Slides for lecture #2"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/feedforward-neural-network-ii-multi-class-classification
    image_path: /assets/images/ffnn-multi2.png
    alt: "Slides for lecture #2"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/feedforward-neural-network-ii-multi-class-classification
    image_path: /assets/images/ffnn-multi3.png
    alt: "Slides for lecture #2"
    title: "Open on Speaker Deck"
lecture3:
  - url: https://speakerdeck.com/chokkan/word-embeddings
    image_path: /assets/images/wordemb.png
    alt: "Slides for lecture #3"
    title: "Open on Speaker Deck"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_en.ipynb
    image_path: /assets/images/word2vec_en.png
    alt: "Word vectors pre-trained on English newspapers"
    title: "Word vectors pre-trained on English newspapers"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_ja.ipynb
    image_path: /assets/images/word2vec_ja.png
    alt: "Word vectors trained on Japanese Wikipedia"
    title: "Word vectors trained on Japanese Wikipedia"
lecture4:
  - url: https://speakerdeck.com/chokkan/dnn-for-structural-data
    image_path: /assets/images/structure1.png
    alt: "Slides for lecture #4"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/dnn-for-structural-data
    image_path: /assets/images/structure2.png
    alt: "Slides for lecture #4"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/dnn-for-structural-data
    image_path: /assets/images/structure3.png
    alt: "Slides for lecture #4"
    title: "Open on Speaker Deck"
lecture5:
  - url: https://speakerdeck.com/chokkan/encoder-decoder-models
    image_path: /assets/images/encdec1.png
    alt: "Slides for lecture #5"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/encoder-decoder-models
    image_path: /assets/images/encdec2.png
    alt: "Slides for lecture #5"
    title: "Open on Speaker Deck"
  - url: https://speakerdeck.com/chokkan/encoder-decoder-models
    image_path: /assets/images/encdec3.png
    alt: "Slides for lecture #5"
    title: "Open on Speaker Deck"
---

{% include feature_row id="intro" type="center" %}

# Lecture #1: Feedforward Neural Network (I)

{% include gallery id="lecture1" caption="Keywords: binary classification, Threshold Logic Units (TLUs), Single-layer Perceptron (SLP), Perceptron algorithm, sigmoid function, Stochastic Gradient Descent (SGD), Multi-layer Perceptron (MLP), Backpropagation, Computation Graph, Automatic Differentiation, Universal Approximation Theorem. Interactive demos available for [single-layer perceptron](demo-slp.html) and [multi-layer perceptron](demo-mlp.html). Implementation available on [github](https://github.com/chokkan/deeplearning/blob/master/notebook/binary.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/binary.ipynb)." %}

# Lecture #2: Feedforward Neural Network (II)

{% include gallery id="lecture2" caption="Keywords: multi-class classification, linear multi-class classifier, softmax function, Stochastic Gradient Descent (SGD), mini-batch training, loss functions, activation functions, dropout. All the above images link to the same slides. Implementation available on [github](https://github.com/chokkan/deeplearning/blob/master/notebook/mnist.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/mnist.ipynb)." %}

# Lecture #3: Word embeddings

{% include gallery id="lecture3" caption="Keywords: word embeddings, distributed representation, distributional hypothesis, pointwise mutual information, singular value decomposition, word2vec, word analogy, GloVe, fastText. Demo with word vectors trained on [Japanese Wikipedia](https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_ja.ipynb) and [English newspapers](https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_en.ipynb) available." %}

# Lecture #4: DNN for structural data

{% include gallery id="lecture4" caption="Keywords: Recurrent Neural Networks (RNNs), Gradient vanishing and exploding, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), Recursive Neural Network, Tree-structured LSTM, Convolutional Neural Networks (CNNs). All the above images link to the same slides. Implementation available on [Github](https://github.com/chokkan/deeplearning/blob/master/notebook/rnn.ipynb) and [Colaboratory](https://colab.research.google.com/github/chokkan/deeplearning/blob/master/notebook/rnn.ipynb)." %}

# Lecture #5: Encoder-decoder models

{% include gallery id="lecture5" caption="Keywords: language modeling, Recurrent Neural Network Language Model (RNNLM), encoder-decoder models, sequence-to-sequence models, attention mechanism, reading comprehension, question answering, headline generation, multi-task learning, character-based RNN, byte-pair encoding, Convolutional Sequence to Sequence (ConvS2S), Transformer, coverage. All the above images link to the same slides." %}
