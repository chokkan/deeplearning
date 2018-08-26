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
    excerpt: "Slides for lecture #1"
  - url: demo-slp.html
    image_path: /assets/images/demo-slp.png
    alt: "Interactive single-layer perceptron"
    excerpt: "Interactive demo of single-layer perceptron"
  - url: demo-mlp.html
    image_path: /assets/images/demo-mlp.png
    alt: "Interactive multi-layer perceptron"
    excerpt: "Interactive demo of multi-layer perceptron"
  - url: binary-impl.html
    image_path: /assets/images/binary-impl.png
    alt: "Implementations"
    excerpt: "Implementations in PyTorch, Chanier, TensorFlow, MXNet"    
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/binary.ipynb
    image_path: /assets/images/pytorch.png
    alt: "Implementations in Jupyter notebook (Google Colaboratory)"
    excerpt: "The same implementations in Jupyter notebook (Google Colaboratory)"    
lecture2:
  - url: https://speakerdeck.com/chokkan/feedforward-neural-network-ii-multi-class-classification
    image_path: /assets/images/ffnn-multi2.png
    alt: "Slides for lecture #2"
    excerpt: "Slides for lecture #2"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/mnist.ipynb
    image_path: /assets/images/multi-impl.png
    alt: "Implementation in Jupyter notebook (Google Colaboratory)"
    excerpt: "Implementation in Jupyter notebook (Google Colaboratory)"
lecture3:
  - url: https://speakerdeck.com/chokkan/word-embeddings
    image_path: /assets/images/wordemb.png
    alt: "Slides for lecture #3"
    excerpt: "Slides for lecture #3"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_en.ipynb
    image_path: /assets/images/word2vec_en.png
    alt: "Word vectors pre-trained on English newspapers"
    excerpt: "Word vectors pre-trained on English newspapers"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/word2vec_ja.ipynb
    image_path: /assets/images/word2vec_ja.png
    alt: "Word vectors trained on Japanese Wikipedia"
    excerpt: "Word vectors trained on Japanese Wikipedia"
lecture4:
  - url: https://speakerdeck.com/chokkan/dnn-for-structural-data
    image_path: /assets/images/structure3.png
    alt: "Slides for lecture #4"
    excerpt: "Slides for lecture #4"
  - url: https://github.com/chokkan/deeplearning/blob/master/notebook/rnn.ipynb
    image_path: /assets/images/structure-impl.png
    alt: "Implementation in Jupyter notebook (Google Colaboratory)"
    excerpt: "Implementation in Jupyter notebook (Google Colaboratory)"
lecture5:
  - url: https://speakerdeck.com/chokkan/encoder-decoder-models
    image_path: /assets/images/encdec3.png
    alt: "Slides for lecture #5"
    except: "Slides for lecture #5"
---

{% include feature_row id="intro" type="center" %}

# Lecture #1: Feedforward Neural Network (I)

Keywords: binary classification, Threshold Logic Units (TLUs), Single-layer Perceptron (SLP), Perceptron algorithm, sigmoid function, Stochastic Gradient Descent (SGD), Multi-layer Perceptron (MLP), Backpropagation, Computation Graph, Automatic Differentiation, Universal Approximation Theorem.

{% include features id="lecture1" type="center" %}

# Lecture #2: Feedforward Neural Network (II)

Keywords: multi-class classification, linear multi-class classifier, softmax function, Stochastic Gradient Descent (SGD), mini-batch training, loss functions, activation functions, dropout.

{% include features id="lecture2" type="center" %}

# Lecture #3: Word embeddings

Keywords: word embeddings, distributed representation, distributional hypothesis, pointwise mutual information, singular value decomposition, word2vec, word analogy, GloVe, fastText.

{% include features id="lecture3" type="center" %}

# Lecture #4: DNN for structural data

Keywords: Recurrent Neural Networks (RNNs), Gradient vanishing and exploding, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), Recursive Neural Network, Tree-structured LSTM, Convolutional Neural Networks (CNNs).

{% include features id="lecture4" type="center" %}

# Lecture #5: Encoder-decoder models

Keywords: language modeling, Recurrent Neural Network Language Model (RNNLM), encoder-decoder models, sequence-to-sequence models, attention mechanism, reading comprehension, question answering, headline generation, multi-task learning, character-based RNN, byte-pair encoding, Convolutional Sequence to Sequence (ConvS2S), Transformer, coverage.

{% include features id="lecture5" type="center" %}
