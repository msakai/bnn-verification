# BNN verification dataset for Max-SAT Evaluation 2020

## Usage

(TBD)

## Example

## `bnn_mnist_rot_10_label4_adversarial_norm_inf_totalizer.wcnf`

| | Image | Prediction of a model | P(y=0)<br>(logit) | P(y=1)<br>(logit) | P(y=2)<br>(logit) | P(y=3)<br>(logit) | P(y=4)<br>(logit) | P(y=5)<br>(logit) | P(y=6)<br>(logit) | P(y=7)<br>(logit) | P(y=8)<br>(logit) | P(y=9)<br>(logit) |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Original image | ![original](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_orig.png) | 4 | 3.1416737e-14<br>(8.883254) | 5.5133663e-22<br>(-8.975005) | 1.2148612e-05<br>(28.656395) | 7.593513e-20<br>(-4.049718) | **0.9997013**<br>(**39.974392**) | 1.711211e-12<br>(12.88087) | 3.8705436e-10<br>(18.302235) | 0.00028651825<br>(31.816982) | 5.633235e-12<br>(14.072353) | 4.0916482e-11<br>(16.055202) |
| Pertubated image) | ![perturbated](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_perturbated.png) | 6 | 4.5545687e-10<br>(12.883254) | 2.6813108e-21<br>(-12.975005) | 0.0032257813<br>(28.656395) | 1.7916893e-10<br>(11.950282) | 0.0016309624<br>(27.97439) | 0.004037595<br>(28.880869) | **0.91325474**<br>(**34.302235**) | 0.07607825<br>(31.816982) | 4.4588405e-06<br>(22.072353) | 0.0017682364<br>(28.055202) |

Added perturbation:

* 0-norm: 18.0
* 1-norm: 18.0
* 2-norm: 4.242640687119285
* ∞-norm: 1.0

## Submission to Max-SAT evaluation 2020

* [Description](maxsat2020/description.pdf)
* [Submitted instances](https://www.dropbox.com/s/s5r30rcpfby1vmd/maxsat2020_bnn_verification.tar.gz?dl=0) (29.62 GB)
  * [Problems actually used in the competition](https://www.dropbox.com/scl/fi/o5iseq0pm4ynsi3oq5d2m/maxsat2020_bnn_verification_used.tar.gz?rlkey=brvvfdxs0v4o56f9vo29bvskk&dl=0) (2.5 GB)

## Talk at NII Shonan Meeting No. 180 “The Art of SAT”

* [Program](https://nikolajbjorner.github.io/ShonanArtOfSAT/program.html)
* [Slides](https://nikolajbjorner.github.io/ShonanArtOfSAT/MasahiroSakai-slides.pdf)

### Some follow-ups

* Q: In several samples used in the contest, the images do not look like the numbers shown on the labels
  * A: This problem was caused by my misunderstanding of the order of the features in `MNIST-rot` and `MNIST-back-image` datasets (`MNIST` does not have this problem). Thereby images were rotated and flipped from their original form. This problem should have been resolved in the preprocessing during data set creation. However, this is a visualization-only issue, since training and inference treat data in a consistent manner.

## References

* [MaxSAT Evaluation 2020 : Solver and Benchmark Descriptions](https://helda.helsinki.fi/items/a24cd636-edb1-4e20-bbdf-e56a66a3a05c)
