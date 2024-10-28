# BNN verification instances for MIPLIB 2024 submission

This dataset consists of MILP instances for finding minimal perturbation adversarial examples of BNNs (binarized neural networks).

The authors have previously submitted similar problem instances to [Max-SAT Evaluation 2020](https://maxsat-evaluations.github.io/2020/) [1], and this is its MILP version. Detailed information including source code is available at <https://github.com/msakai/bnn-verification/>.

## Problem overview

Given a trained neural network f and an input x⁰, the goal is to find the minimal perturbation ε such that x⁰ + ε is misclassified, i.e. we consider the following optimization problem.

```
minimize ǁεǁ

subject to f(x⁰ + ε) ≠ f(x⁰)
```

In our case, the task is hand-written digit classification. The input space is the 8-bit image of 28×28 (= 784) pixels and the output space is {0,…,9}.

```
f: {0,…,255}⁷⁸⁴ → {0,…,9}
```

## Target neural networks

The network architecture is based on BNNs (binarized neural networks).

We omit the detail of BNN here, but our BNN consists of the following steps:
1. Each input pixel xᵢ ∈ {0, …, 255} is first binarized as zᵢ = binᵢ(xᵢ) ∈ {-1, +1} using learned threshold (note that threshold varies depending on i),
2. Some computation g is applied to z to obtain logits = g(z) ∈ R¹⁰,
3. Then, y = argmaxⱼ logitsⱼ is the output.

```
f = argmax ∘ g ∘ bin
```

We trained BNNs on three datasets: [MNIST](https://yann.lecun.com/exdb/mnist/) and its two variants [MNIST-rot and MNIST-back-image](http://web.archive.org/web/20180831072509/http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations).

## Objective functions

We consider four norms: L₀, L₁, L₂ and L<sub>∞</sub>. (In the Max-SAT Evaluation 2020, we were able to submit only L<sub>∞</sub> cases, but this time we have prepared L₀, L₁, L₂  cases too.)

## Problem Instances

We use the following five images. These are the images used in the five problems selected for the Max-SAT Evaluation 2020.

|Dataset|Instance No.|Image|True Label|
|-|-:|-|-:|
|MNIST|7|![](images/bnn_mnist_7_label9.png)|9|
|MNIST-rot|8|![](images/bnn_mnist_rot_8_label1.png)|1|
|MNIST-rot|16|![](images/bnn_mnist_rot_16_label5.png)|5|
|MNIST-back-image|32|![](images/bnn_mnist_back_image_32_label3.png)|3|
|MNIST-back-image|73|![](images/bnn_mnist_back_image_73_label5.png)|5|


With the five images and four norm combinations, there are 20 problems in total.

The file names of problem instances are in the following form:

```
bnn_{dataset_name}_{instance number}_label{true label}_adversarial_norm_{norm's p}.lp
```

## Some notes on MILP encoding

### Decision variables

We use `input_bin(i)`s instead of εᵢs as decision variables.

`input_bin(i)` ∈ {0, 1} corresponds to (binᵢ(x⁰ᵢ + εᵢ) + 1) / 2.

Conversely, let wᵢ be the smallest magnitude perturbation to flip binᵢ(xᵢ), i.e. binᵢ(x⁰ᵢ + wᵢ) ≠ binᵢ(x⁰ᵢ) and binᵢ(x⁰ᵢ + v) = binᵢ(x⁰ᵢ) for all v such that |v| < |wᵢ|. Then we can reconstruct εᵢ as wᵢ I[`input_bin(i)` ≠ (binᵢ(x⁰ᵢ) + 1) / 2].

### Output variables

`output(j)`s are one hot encoding of f(x + ε) ∈ {0,…,9}.

### Objective functions

L<sub>∞</sub>-norm objective function is ǁεǁ<sub>∞</sub> = max {|εᵢ|}ᵢ = max {|wᵢ| I[`input_bin(i)` ≠ (binᵢ(x⁰ᵢ) + 1) / 2]}ᵢ. This can be minimized by minimizing a fresh variable u under the constraints |wᵢ| I[`input_bin(i)` ≠ (binᵢ(x⁰ᵢ) + 1) / 2] ≤ u for all i.

(We used more complicated encoding in Max-SAT evaluation to encode the problem as Max-SAT problems [1], but here we use the one that is simple and is standard in MILP.)

For Lₚ-norm cases (p ≠ ∞), minimizing ǁεǁₚ is equivalent to minimizing ǁεǁₚᵖ = ∑ᵢ |wᵢ|ᵖ I[input_bin(i) ≠ (binᵢ(x⁰ᵢ) + 1) / 2]. We use the last expression as the objective function in our MILP encoding.

## Known solutions

For L<sub>∞</sub>-norm cases, we already know optimal solutions.

|Problem instance|Solution| ǁεǁ<sub>∞</sub>|Original Image|Predicted Label|Perturbated Image<sup>†</sup>|Predicted Label|
|-|-|-:|-|-:|-|-:|
|bnn_mnist_7_label9_adversarial_norm_inf.lp.bz2|[solution](solutions/bnn_mnist_7_label9_adversarial_norm_inf.sol)|1|![](images/bnn_mnist_7_label9.png)|9|![](solutions/bnn_mnist_7_label9_adversarial_norm_inf.png)|5|
|bnn_mnist_rot_8_label1_adversarial_norm_inf.lp.bz2|[soltuion](solutions/bnn_mnist_rot_8_label1_adversarial_norm_inf.sol)|1|![](images/bnn_mnist_rot_8_label1.png)|1|![](solutions/bnn_mnist_rot_8_label1_adversarial_norm_inf.png)|3|
|bnn_mnist_rot_16_label5_adversarial_norm_inf.lp.bz2|[soltuion](solutions/bnn_mnist_rot_16_label5_adversarial_norm_inf.sol)|1|![](images/bnn_mnist_rot_16_label5.png)|5|![](solutions/bnn_mnist_rot_16_label5_adversarial_norm_inf.png)|7|
|bnn_mnist_back_image_32_label3_adversarial_norm_inf.lp.bz2|[soltuion](solutions/bnn_mnist_back_image_32_label3_adversarial_norm_inf.sol)|2|![](images/bnn_mnist_back_image_32_label3.png)|3|![](solutions/bnn_mnist_back_image_32_label3_adversarial_norm_inf.png)|8|
|bnn_mnist_back_image_73_label5_adversarial_norm_inf.lp.bz2|[soltuion](solutions/bnn_mnist_back_image_73_label5_adversarial_norm_inf.sol)|4|![](images/bnn_mnist_back_image_73_label5.png)|5|![](solutions/bnn_mnist_back_image_73_label5_adversarial_norm_inf.png)|3|

For other norm cases, however, optimal solutions are not yet known.

## References

* [1] M. Sakai “BNN verification dataset for Max-SAT Evaluation 2020,”
  In MaxSAT Evaluation 2020: Solver and Benchmark Descriptions. 2020,
  pp. 37-28. <http://hdl.handle.net/10138/318451>
