---
layout: single
permalink: demo-mlp:output_ext
title: Interactive multi-layer perceptron
sidebar:
  - title: "Parameters"
    text: |
      <table>
        <tr>
          <td>$f_h^{(1)}$</td>
          <td colspan="2">
            <select id="f1h">
              <option value="none" selected>None</option>
              <option value="step">Step</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="relu">ReLU</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>$w_{hx}^{(1)}$</td>
          <td><input id="w1hx" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w1hx_v">1.0</span></td>
        </tr>
        <tr>
          <td>$w_{hy}^{(1)}$</td>
          <td><input id="w1hy" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w1hy_v">1.0</span></td>
        </tr>
        <tr>
          <td>$b_h^{(1)}$</td>
          <td><input id="b1h" type="range" min="-10" max="10" step="0.1" value="0"></td>
          <td><span id="b1h_v">0.0</span></td>
        </tr>
        <tr>
          <td>$f_v^{(1)}$</td>
          <td colspan="2">
            <select id="f1v">
              <option value="none" selected>None</option>
              <option value="step">Step</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="relu">ReLU</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>$w_{vx}^{(1)}$</td>
          <td><input id="w1vx" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w1vx_v">1.0</span></td>
        </tr>
        <tr>
          <td>$w_{vy}^{(1)}$</td>
          <td><input id="w1vy" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w1vy_v">1.0</span></td>
        </tr>
        <tr>
          <td>$b_v^{(1)}$</td>
          <td><input id="b1v" type="range" min="-10" max="10" step="0.1" value="0"></td>
          <td><span id="b1v_v">0.0</span></td>
        </tr>
        <tr>
          <td>$f_z^{(2)}$</td>
          <td colspan="2">
            <select id="f2z">
              <option value="none" selected>None</option>
              <option value="step">Step</option>
              <option value="sigmoid">Sigmoid</option>
              <option value="tanh">Tanh</option>
              <option value="relu">ReLU</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>$w_{zh}^{(2)}$</td>
          <td><input id="w2zh" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w2zh_v">1.0</span></td>
        </tr>
        <tr>
          <td>$w_{zv}^{(2)}$</td>
          <td><input id="w2zv" type="range" min="-10" max="10" step="0.1" value="1"></td>
          <td><span id="w2zv_v">1.0</span></td>
        </tr>
        <tr>
          <td>$b_z^{(2)}$</td>
          <td><input id="b2z" type="range" min="-10" max="10" step="0.1" value="0"></td>
          <td><span id="b2z_v">0.0</span></td>
        </tr>
      </table>

---

<div id="heatmap"></div>

\\[
 h = f_h^{(1)}(w_{hx}^{(1)} x + w_{hy}^{(1)} y + b_h^{(1)})
\\]
\\[
 v = f_v^{(1)}(w_{vx}^{(1)} x + w_{vy}^{(1)} y + b_v^{(1)})
\\]
\\[
 z = f_z^{(2)}(w_{zh}^{(2)} h + w_{zv}^{(2)} v + b_z^{(2)})
\\]

This page visualizes a multi-layer perceptron with two inputs $x$ and $y$, two hidden units $h$ and $v$, and one output $z$.

Here:

+ $w_{hx}^{(1)}$, $w_{hy}^{(1)}$, $b_h^{(1)}$, $f_h^{(1)}$ denote two weights, a bias, and an activation function for computing $h$ from the inputs $x$ and $y$;
+ $w_{vx}^{(1)}$, $w_{vy}^{(1)}$, $b_v^{(1)}$, $f_v^{(1)}$ denote two weights, a bias, and an activation function for computing $v$ from the inputs $x$ and $y$;
+ $w_{zh}^{(2)}$, $w_{zv}^{(2)}$, $b_z^{(2)}$, $f_z^{(2)}$ denote two weights, a bias, and an activation function for computing $z$ from the hidden units $h$ and $v$;

In this visualization, one can see outputs from the two-layer perceptron as a *heat map* as they *interactively change* the parameters in the perceptron. The heatmap represents two inputs $x$ and $y$ as $x$-axis and $y$-axis, respectively, and an output as color thickness of the plotting area.

Let's confirm that changing the parameters of the perceptron can realize XOR.

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
$('select').on("change", function() { update(); });
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
    t: 40
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

function getf(name)
{
  console.log(name);
  var f = fnone;
  switch (name) {
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
  return f;
}

function buildData(
  f1h, w1hx, w1hy, b1h,
  f1v, w1vx, w1vy, b1v,
  f2z, w2zh, w2zv, b2z)
{
  var zValues = [];
  for (y = y_min; y <= y_max; y += step) {
    values = [];
    for (x = x_min; x <= x_max; x += step) {
      h = f1h(w1hx * x + w1hy * y + b1h);
      v = f1v(w1vx * x + w1vy * y + b1v);
      z = f2z(w2zh * h + w2zv * v + b2z);
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
  var f1h = getf($('#f1h').val());
  var w1hx = parseFloat($('#w1hx').val());
  var w1hy = parseFloat($('#w1hy').val());
  var b1h = parseFloat($('#b1h').val());
  var f1v = getf($('#f1v').val());
  var w1vx = parseFloat($('#w1vx').val());
  var w1vy = parseFloat($('#w1vy').val());
  var b1v = parseFloat($('#b1v').val());
  var f2z = getf($('#f2z').val());
  var w2zh = parseFloat($('#w2zh').val());
  var w2zv = parseFloat($('#w2zv').val());
  var b2z = parseFloat($('#b2z').val());

  $('#w1hx_v').html(w1hx.toFixed(1));
  $('#w1hy_v').html(w1hy.toFixed(1));
  $('#b1h_v').html(b1h.toFixed(1));
  $('#w1vx_v').html(w1vx.toFixed(1));
  $('#w1vy_v').html(w1vy.toFixed(1));
  $('#b1v_v').html(b1v.toFixed(1));
  $('#w2zh_v').html(w2zh.toFixed(1));
  $('#w2zv_v').html(w2zv.toFixed(1));
  $('#b2z_v').html(b2z.toFixed(1));

  var data = buildData(
    f1h, w1hx, w1hy, b1h,
    f1v, w1vx, w1vy, b1v,
    f2z, w2zh, w2zv, b2z);
  Plotly.newPlot('heatmap', data, layout);  
}

update();
</script>
