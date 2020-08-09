---
layout: splash
header:
  overlay_color: "#000"
  overlay_filter: "0.7"
  og_image: /assets/images/intro-to-dl.png
  overlay_image: /assets/images/splash.png
excerpt: 'with flavor of Natural Language Processing (NLP)'
description: "Deep Neural Networks (FFNN, CNN, RNN, LSTM, GRU), Word Embeddings, Encoder-Decoder (Attention, Transformer, GPT, BERT)."
intro: 
  - excerpt: 'This site accompanies the latter half of the [ART.T458: Advanced Machine Learning](http://www.ocw.titech.ac.jp/index.php?module=General&action=T0300&GakubuCD=4&GakkaCD=342305&KeiCD=23&course=5&KougiCD=202004845&Nendo=2020&lang=EN&vid=03) course at [Tokyo Institute of Technology](https://www.titech.ac.jp/english/), which focuses on Deep Learning for Natural Language Processing (NLP).'
lecture1:
  - title: "Slides"
    speakerdeck:
      id: 932bbd2dad6b45eaac6c4ad0270740c4
      ratio: 1.44428772919605
  - url: demo-slp.html
    image_path: /assets/images/demo-slp.png
    alt: "Interactive single-layer perceptron"
    title: "Interactive SLP model"
  - url: demo-mlp.html
    image_path: /assets/images/demo-mlp.png
    alt: "Interactive multi-layer perceptron"
    title: "Interactive MLP model"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/binary.ipynb
    image_path: /assets/images/pytorch.png
    alt: "Implementations in Jupyter notebook"
    title: "Implementations"
    excerpt: "Perceptron algorithm in numpy; automatic differentiation in autograd, pytorch, TensorFlow, and JAX; single and multi layer neural network in pytorch."
lecture2:
  - title: "Slides"
    speakerdeck:
      id: a37a47297ebe447f980eeafe7cdcdd7f
      ratio: 1.44428772919605
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/mnist.ipynb
    image_path: /assets/images/multi-impl.png
    alt: "Implementations in Jupyter notebook"
    title: "Implementations"
    excerpt: "Preparing the MNIST dataset; perceptron algorithm in numpy; stochastic gradient descent in numpy; single and multi layer neural network in pytorch."
lecture3:
  - title: "Slides"
    speakerdeck:
      id: 86260627cf1e41e8876e2a0e581c4670
      ratio: 1.44428772919605
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/resnet.ipynb
    image_path: /assets/images/resnet.png
    alt: "Using ResNet-50"
    title: "Object recognition"
    excerpt: "Classify an image using ResNet-50."
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/convolution.ipynb
    image_path: /assets/images/convolution.png
    alt: "Convolutions as Image filters"
    title: "Image filters"
    excerpt: "Various image filters by manually setting values of a weight matrix in torch.nn.Conv2d."
lecture4:
  - title: "Slides"
    speakerdeck:
      id: bb5ae4e2e81a453699c453d1db158b28
      ratio: 1.44428772919605
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_en.ipynb
    image_path: /assets/images/word2vec_en.png
    alt: "Loading word vectors pre-trained on English newspapers; computing similarity; word analogy"
    excerpt: "Loading word vectors pre-trained on English news; computing similarity; word analogy"
    title: "English Word Vector"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_ja.ipynb
    image_path: /assets/images/word2vec_ja.png
    alt: "Loading word vectors trained on Japanese Wikipedia; computing similarity; word analogy"
    excerpt: "Loading word vectors trained on Japanese Wikipedia; computing similarity; word analogy"
    title: "Japanese Word Vector"
lecture5:
  - title: "Slides"
    speakerdeck:
      id: 13c7414498d843b5ae7d539a14f66f50
      ratio: 1.44428772919605
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/rnn.ipynb
    image_path: /assets/images/structure-impl.png
    alt: "Implementations in Jupyter notebook"
    title: "Implementations"
    excerpt: "RNN; Mini-batch RNN"
lecture6:
  - title: "Slides"
    speakerdeck:
      id: e9af4583e098484881d8259647ffbb5d
      ratio: 1.44428772919605
---

{% include feature_row id="intro" type="center" %}

# Lecture #1: Feedforward Neural Network (I)

Keywords: binary classification, Threshold Logic Units (TLUs), single-layer neural network, Perceptron algorithm, sigmoid function, stochastic gradient descent (SGD), multi-layer neural network, backpropagation, computation graph, automatic differentiation, universal approximation theorem.

{% include feature_row_custom id="lecture1" %}

# Lecture #2: Feedforward Neural Network (II)

Keywords: multi-class classification, linear multi-class classifier, softmax function, stochastic gradient descent (SGD), mini-batch training, loss functions, activation functions, ReLU, dropout.

{% include feature_row_custom id="lecture2" %}

# Lecture #3: Convolutional Neural Network

Keywords: Convolutional Neural Networks (CNNs), MNIST, 2D convolution, padding, stride, channels, image filter, max pooling, ILSVRC, ImageNet, AlexNet, VGGNet, ResNet.

{% include feature_row_custom id="lecture3" %}

# Lecture #4: Word embeddings

Keywords: word embeddings, distributed representation, distributional hypothesis, pointwise mutual information, singular value decomposition, word2vec, word analogy, GloVe, fastText.

{% include feature_row_custom id="lecture4" %}

# Lecture #5: DNN for structural data

Keywords: Recurrent Neural Networks (RNNs), Gradient vanishing and exploding, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), Recursive Neural Network, Tree-structured LSTM, Convolutional Neural Networks (CNNs).

{% include feature_row_custom id="lecture5" %}

# Lecture #6: Encoder-decoder models

Keywords: language modeling, Recurrent Neural Network Language Model (RNNLM), encoder-decoder models, sequence-to-sequence models, attention mechanism, Convolutional Sequence to Sequence (ConvS2S), Transformer, GPT, BERT.

{% include feature_row_custom id="lecture6" %}

<script type="text/javascript" >
  window.onload = function () {
    $(window).trigger('resize');
  }
</script>
