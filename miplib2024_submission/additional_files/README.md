# BNN verification instances for MIPLIB 2024 submission

This dataset consists of MILP instances for finding minimal perturbation adversarial examples of BNNs (binarized neural networks).

The authors have previously submitted similar problem instances to Max-SAT Evaluation 2020 [1], and this is its MILP version.

## Problem overview

Given a trained neural network $f$ and input $x^0$, the goal is to find the minimal perturbation $\epsilon$ such that $x^0 + \epsilon$ is misclassified, i.e. we consider the following optimization problem.

$$
\begin{align*}
&\underset{\epsilon}{\text{minimize}}& & \| x^0 \|_p \\
&\text{subject\;to}
& & f(x^0 + \epsilon) \ne f(x^0) \\
\end{align*}
$$

## Input images: $x^0$

Our task is hand-written digit classification, and we use the following five images. These are the images used in the five problems used in the Max-SAT Evaluation 2020.

|Dataset|Instance No.|Image|True Label|
|-|-:|-|-:|
|MNIST|7|![](images/bnn_mnist_7_label9.png)|9|
|MNIST-rot|8|![](images/bnn_mnist_rot_8_label1.png)|1|
|MNIST-rot|16|![](images/bnn_mnist_rot_16_label5.png)|5|
|MNIST-back-image|32|![](images/bnn_mnist_back_image_32_label3.png)|3|
|MNIST-back-image|73|![](images/bnn_mnist_back_image_73_label5.png)|5|

They are 8-bit image of $28\times 28$ pixels and represented as 8-bit 784 ($= 28\times 28$) dimension vectors (i.e. $x \in \{0, \ldots, 255\} ^{784}$).

## Target neural networks: $f$

The network architecture is based on BNNs (binarized neural networks) and the networks are trained on three datasets *MNIST*, *MNIST-rot*, *MNIST-back-image*.

The representation of neural networks in MILP is similar to the one described in [1], but simpler. This is because (conditional) cardinality constraints do not need to be *encoded* into SAT-level constraints, but can be used as-is as linear constraints.

In our BNNs, each input pixel $x_i \in \{0, \ldots, 255\}$ is first binalized as $z_i = \text{bin}_i(x_i) \in \{-1, +1\}$ using learnt threashold (the threshold depends on $i$). Since it is easy to construct $x$ from $z$ that is closest to $x^0$, we use $z_i$ s instead of $x_i$ s as decision variables in the following.

## Objective functions

We consider four norm: $L_0$, $L_1$, $L_2$ and $L_\infty$.

Let $w_i$ be the smallest change to flip $\text{bin}_i(x_i)$, i.e. $\text{bin}_i(x^0_i + w_i) \ne \text{bin}_i(x^0_i)$.

Then $L_\infty$-norm objective is $\max \{|w_i| I(z_i \ne \text{bin}_i(x^0_i))\}_i$. This can be optimized by introducing a fresh variable $u$:

$$
\begin{align*}
&\underset{z}{\text{minimize}}& & u \\
&\text{subject\;to}
& & |w_i| I(z_i \ne \text{bin}_i(x^0_i)) \le u & \text{for\ all} i\\
& & & \cdots
\end{align*}
$$

(We used more complex encoding in Max-SAT evaluation to encode the problem as Max-SAT problems [1], but here we the one that is standard in MILP.)

For $L_p$-norm ($p \ne \infty$) case, we solve the following problems:

$$
\begin{align*}
&\underset{z}{\text{minimize}}& & \sum_i |w_i|^p I(z_i \ne \text{bin}_i(x^0_i))  \\
&\text{subject\;to}
& & \cdots
\end{align*}
$$

 We were able to submit only $L_\infty$ cases to Max-SAT Evaluation 2020 due to the lack of time, but this time we have prepared $L_0$, $L_1$, $L_2$ cases too.

## Problem Instances

With the above four images and four norm combinations, there are 20 problems.

File names are in the following form.

```
bnn_{dataset_name}_{instance number}_label{true label}_ adversarial_norm_{norm's p}.wcnf
```

## Known solutions

For $L_\infty$-norm cases, we already know optimal solutions.

|Problem instance|Solution|Minimum ǁεǁ<sub>∞</sub>|Original Image|Predicted Label|Perturbated Image<sup>†</sup>|Predicted Label|
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
