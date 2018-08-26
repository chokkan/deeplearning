---
layout: splash
permalink: binary-impl:output_ext
title: Implementations for binary classifiers
---

# Feedforward Neural Networks

This page explains various ways of implementing single-layer and multi-layer neural networks as a supplementary material of [this lecture](https://speakerdeck.com/chokkan/feedforward-neural-network-i-binary-classification). The implementations appear in explicit to abstract order so that one can understand the black-boxed internal processing in deep learning frameworks.

In order to focus on the internals, this page uses a simple and classic example: *threshold logic units*.
Supposing $x=0$ as *false* and $x=1$ as *true*, single-layer neural networks can realize logic units such as AND ($\wedge$), OR ($\vee$), NOT ($\lnot$), and NAND ($|$). Multi-layer neural networks can realize logical compounds such as XOR.

| $x_1$ | $x_2$ | AND | OR | NAND | XOR |
| :---: |:-----:|:---:|:--:|:----:|:---:|
| 0 | 0 | 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 | 0 |

## Single-layer perceptron

We consider a single layer perceptron that predicts a binary label $\hat{y} \in \{0, 1\}$ for a given input vector $\boldsymbol{x} \in \mathbb{R}^d$ ($d$ presents the number of dimensions of inputs) by using the following formula,

$$
\hat{y} = g(\boldsymbol{w} \cdot \boldsymbol{x} + b) = g(w_1 x_1 + w_2 x_2 + ... + w_d x_d + b)
$$

Here, $\boldsymbol{w} \in \mathbb{R}^d$ is a weight vector; $b \in \mathbb{R}$ is a bias weight; and $g(.)$ denotes a Heaviside step function (we assume $g(0)=0$).

Let's train a NAND gate with two inputs ($d = 2$). More specifically, we want to find a weight vector $\boldsymbol{w}$ and a bias weight $b$ of a single-layer perceptron that realizes the truth table of the NAND gate: $\\{0,1\\}^2 \to \\{0,1\\}$.

We convert the truth table into a training set consisting of all mappings of the NAND gate,

$$
\boldsymbol{x}_1 = (0, 0), y_1 = 1 \\
\boldsymbol{x}_2 = (0, 1), y_2 = 1 \\
\boldsymbol{x}_3 = (1, 0), y_3 = 1 \\
\boldsymbol{x}_4 = (1, 1), y_4 = 0 \\
$$

In order to train a weight vector and bias weight in a unified code, we include a bias term as an additional dimension to inputs. More concretely, we append $1$ to each input,

$$
\boldsymbol{x}'_1 = (0, 0, 1), y_1 = 1 \\
\boldsymbol{x}'_2 = (0, 1, 1), y_2 = 1 \\
\boldsymbol{x}'_3 = (1, 0, 1), y_3 = 1 \\
\boldsymbol{x}'_4 = (1, 1, 1), y_4 = 0 \\
$$

Then, the formula of the single-layer perceptron becomes,

$$
\hat{y} = g((w_1, w_2, w_3) \cdot \boldsymbol{x}') = g(w_1 x_1 + w_2 x_2 + w_3)
$$

In other words, $w_1$ and $w_2$ present weights for $x_1$ and $x_2$, respectively, and $w_3$ does a bias weight.

The code below implements Rosenblatt's perceptron algorithm with a fixed number of iterations (100 times). We use a constant learning rate 0.5 for simplicity.

{% include notebook/binary/slp_rosenblatt.md %}

## Single-layer perceptron with mini-batch

It is better to reduce the execusion run by the Python interpreter, which is relatively slow. The common technique to speed up a machine-learning code written in Python is to to execute computations within the matrix library (e.g., numpy).

The single-layer perceptron makes predictions for four inputs,

$$
\hat{y}_1 = g(\boldsymbol{x}_1 \cdot \boldsymbol{w}) \\
\hat{y}_2 = g(\boldsymbol{x}_2 \cdot \boldsymbol{w}) \\
\hat{y}_3 = g(\boldsymbol{x}_3 \cdot \boldsymbol{w}) \\
\hat{y}_4 = g(\boldsymbol{x}_4 \cdot \boldsymbol{w}) \\
$$

Here, we define $\hat{Y} \in \mathbb{R}^{4 \times 1}$ and $X \in \mathbb{R}^{4 \times d}$ as,

$$
\hat{Y} = \begin{pmatrix} 
  \hat{y}_1 \\ 
  \hat{y}_2 \\ 
  \hat{y}_3 \\ 
  \hat{y}_4 \\ 
\end{pmatrix},
X = \begin{pmatrix} 
  \boldsymbol{x}_1 \\ 
  \boldsymbol{x}_2 \\ 
  \boldsymbol{x}_3 \\ 
  \boldsymbol{x}_4 \\ 
\end{pmatrix}
$$

Then, we can write the four predictions in one dot-product computation,
$$
\hat{Y} = X \cdot \boldsymbol{w}
$$

The code below implements this idea. The function `np.heaviside()` yields a vector corresponding to the four predictions, applying the step function for every element of the argument.

This technique is frequently used in mini-batch training, where gradients for a small number (e.g., 4 to 128) of instances are computed.

{% include notebook/binary/slp_rosenblatt_batch.md %}

## Stochastic gradient descent (SGD) with mini-batch

Next, we consider a single-layer feedforward neural network with sigmoid activation function.
In essence, we replace Heaviside step function with sigmoid function when predicting $\hat{Y}$ and to use the formula for stochastic gradient descent when updating $\boldsymbol{w}$.

{% include notebook/binary/slp_sgd_numpy.md %}

## Automatic differentiation

### autograd

{% include notebook/binary/ad_autograd.md %}

### PyTorch

{% include notebook/binary/ad_pytorch.md %}

### Chainer

{% include notebook/binary/ad_chainer.md %}

### TensorFlow

{% include notebook/binary/ad_tensorflow.md %}

### MXNet

{% include notebook/binary/ad_mxnet.md %}

## Single-layer neural network using automatic differentiation

{% include code.html tt1="PyTorch" tc1="notebook/binary/slp_ad_pytorch.md" tt2="Chainer" tc2="notebook/binary/slp_ad_chainer.md" tt3="TensorFlow" tc3="notebook/binary/slp_ad_tensorflow.md" tt4="MXNet" tc4="notebook/binary/slp_ad_mxnet.md" %}

## Multi-layer neural network using automatic differentiation

{% include code.html tt1="PyTorch" tc1="notebook/binary/mlp_ad_pytorch.md" tt2="Chainer" tc2="notebook/binary/mlp_ad_chainer.md" tt3="TensorFlow" tc3="notebook/binary/mlp_ad_tensorflow.md" tt4="MXNet" tc4="notebook/binary/mlp_ad_mxnet.md" %}

## Single-layer neural network with high-level NN modules

{% include code.html tt1="PyTorch" tc1="notebook/binary/slp_pytorch_sequential.md" tt2="Chainer" tc2="notebook/binary/slp_chainer_sequential.md" %}

## Multi-layer neural network with high-level NN modules

{% include code.html tt1="PyTorch" tc1="notebook/binary/mlp_pytorch_sequential.md" tt2="Chainer" tc2="notebook/binary/mlp_chainer_sequential.md" %}

## Single-layer neural network with an optimizer

{% include code.html tt1="PyTorch" tc1="notebook/binary/slp_pytorch_sequential_optim.md" tt2="Chainer" tc2="notebook/binary/slp_chainer_sequential_optimizers.md" tt3="TensorFlow" tc3="notebook/binary/slp_tensorflow_keras.md" tt4="MXNet" tc4="notebook/binary/slp_mxnet_sequential_trainer.md" %}

## Multi-layer neural network with an optimizer

{% include code.html tt1="PyTorch" tc1="notebook/binary/mlp_pytorch_sequential_optim.md" tt2="Chainer" tc2="notebook/binary/mlp_chainer_sequential_optimizers.md" tt3="TensorFlow" tc3="notebook/binary/mlp_tensorflow_keras.md" tt4="MXNet" tc4="notebook/binary/mlp_mxnet_trainer.md" %}

## Single-layer neural network with a customizable NN class.

{% include code.html tt1="PyTorch" tc1="notebook/binary/slp_pytorch_class.md" tt2="Chainer" tc2="notebook/binary/slp_chainer_class.md" %}

## Multi-layer neural network with a customizable NN class.

{% include code.html tt1="PyTorch" tc1="notebook/binary/mlp_pytorch_class.md" tt2="Chainer" tc2="notebook/binary/mlp_chainer_class.md" %}

<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js"></script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

<script>
$(document).ready(function() {
  $('pre code[class="language-python"]').each(function(i, block) {
    hljs.highlightBlock(block);
  });
});
</script>