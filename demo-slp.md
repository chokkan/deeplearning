---
layout: single
permalink: demo-slp:output_ext
title: Interactive single-layer perceptron
sidebar:
  - title: "Parameters"
    text: |
      <table>
        <tr>
          <td>$f$</td>
          <td colspan="2">
            <select id="f">
              <option value="none" selected>None</option>
              <option value="step">Step</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="relu">ReLU</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>$w_x$</td>
          <td><input id="wx" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="wxv">1.0</span></td>
        </tr>
        <tr>
          <td>$w_y$</td>
          <td><input id="wy" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="wyv">1.0</span></td>
        </tr>
        <tr>
          <td>$b$</td>
          <td><input id="bias" type="range" min="-10" max="10" step="0.1" value="0"></td>
          <td><span id="biasv">0.0</span></td>
        </tr>
      </table>
      $z = f(w_x x + w_y y + b)$

---

<div id="heatmap"></div>

This page visualizes a single-layer perceptron with two inputs $x$ and $y$ and one output $z$:

\\[
 z = f(w_x x + w_y y + b)
\\]

Here:

+ $w_x$ and $w_y$ denote two weights for the inputs $x$ and $y$, respectively;
+ $b$ is a bias term;
+ $f$ presents an activation function (e.g., sigmoid, tanh, ReLU functions);

In this visualization, one can see outputs from the perceptron as a *heat map* as they *interactively change* the parameters in the perceptron, more concretely, values of $w_x, w_y, b$ and an activation function $f$. The heatmap represents two inputs $x$ and $y$ as $x$-axis and $y$-axis, respectively, and an output as color thickness of the plotting area.

Supposing $0$ as false and $1$ as true, the perceptron can realize logic units (aka. *threshold logic units*) such as AND, OR, and NAND. Let's confirm that changing the parameters of the perceptron realizes these logic units. In addition, experience the reason why a single-layer perceptron cannot realize XOR.

| $x$ | $y$ | AND | OR | NAND | XOR |
| :---: |:-----:|:---:|:--:|:----:|:---:|
| 0 | 0 | 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 | 0 |

<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/rangeslider.js/2.3.2/rangeslider.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

<script>
$('input[type="range"]').rangeslider();
$('#f').on("change", function() { update(); });
$('input[type="range"]').on("input change", function() { update(); });

var x_min = -1.0;
var x_max = +1.0;
var y_min = -1.0;
var y_max = +1.0;
var step = 0.1;

var xValues = [];
for (x = x_min; x <= x_max; x += step)
  xValues.push(x);

var yValues = [];
for (y = y_min; y <= y_max; y += step)
  yValues.push(y);

var layout = {
  height: 600,
  width: 600,
  margin: {
    t: 64
  },
  xaxis: {
    ticks: '',
    title: 'x',
    side: 'bottom',
    hoverformat: '.2f'
  },
  yaxis: {
    ticks: '',
    title: 'y',
    side: 'left',
    hoverformat: '.2f'
  },
  zaxis: {
    hoverformat: '.2f'
  }
};

function fnone(v)
{
  return v;
}

function fstep(v)
{
  return v < 0 ? 0 : 1;
}

function fsigmoid(v)
{
  return 1. / (1. + Math.pow(Math.E, -v));
}

function ftanh(v)
{
  return Math.tanh(v);
}

function frelu(v)
{
  return v < 0 ? 0 : v;
}

function buildData(wx, wy, bias, afunc)
{
  var f = fnone;
  switch (afunc) {
    case "step":
      f = fstep;
      break;
    case "sigmoid":
      f = fsigmoid;
      break;
    case "tanh":
      f = ftanh;
      break;
    case "relu":
      f = frelu;
      break;
  }

  var zValues = [];
  for (y = y_min; y <= y_max; y += step) {
    values = [];
    for (x = x_min; x <= x_max; x += step) {
      z = f(wx * x + wy * y + bias);
      values.push(z);
    }
    zValues.push(values);
  }

  var data = [{
    x: xValues,
    y: yValues,
    z: zValues,
    type: 'heatmap',
  }];

  return data;
}

function update()
{
  var f = $('#f').val();
  var wx = parseFloat($('#wx').val());
  var wy = parseFloat($('#wy').val());
  var bias = parseFloat($('#bias').val());

  $('#wxv').html(wx.toFixed(1));
  $('#wyv').html(wy.toFixed(1));
  $('#biasv').html(bias.toFixed(1));

  var data = buildData(wx, wy, bias, f);
  Plotly.newPlot('heatmap', data, layout);  
}

update();
</script>
