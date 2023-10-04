# BNN verification dataset for Max-SAT Evaluation 2020

## Usage

(TBD)

## Example

## `bnn_mnist_rot_10_label4_adversarial_norm_inf_totalizer.wcnf`

![original](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_orig.png) → ![perturbated](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_perturbated.png)

| | | Prediction of a model | logits[0] | logits[1] | logits[2] | logits[3] | logits[4] | logits[5] | logits[6] | logits[7] | logits[8] | logits[9] |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Original image | ![original](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_orig.png) | 4 | 8.883254 | -8.975005 | 28.656395 | -4.049718 | 39.974392 | 12.88087 | 18.302235 | 31.816982 | 14.072353 | 16.055202 |
| Pertubated image | ![perturbated](./examples/bnn_mnist_rot_10_label4_adversarial_norm_inf_perturbated.png) | 6 | 12.883254 | -12.975005 | 28.656395 | 11.950282 | 27.97439 | 28.880869 | 34.302235 | 31.816982 | 22.072353 | 28.055202 |

Added perturbation:

* 0-norm: 18.0
* 1-norm: 18.0
* 2-norm: 4.242640687119285
* ∞-norm: 1.0

(TODO: Not a good example. Replace it)

## Submission to Max-SAT evaluation 2020

* [Description](maxsat2020/description.pdf)
* [Submitted instances](https://www.dropbox.com/s/s5r30rcpfby1vmd/maxsat2020_bnn_verification.tar.gz?dl=0) (29.62 GB)
  * [Problems actually used in the competition](https://www.dropbox.com/scl/fi/o5iseq0pm4ynsi3oq5d2m/maxsat2020_bnn_verification_used.tar.gz?rlkey=brvvfdxs0v4o56f9vo29bvskk&dl=0) (2.5 GB)

## References

* [MaxSAT Evaluation 2020 : Solver and Benchmark Descriptions](https://helda.helsinki.fi/items/a24cd636-edb1-4e20-bbdf-e56a66a3a05c)
