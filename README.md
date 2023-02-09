# Invariant Collaborative Filtering to Popularity Distribution Shift
Official code of "Invariant Collaborative Filtering to Popularity Distribution Shift" (2023 WWW)

## Overview
InvCF (2022) aims to discover disentangled representations that faithfully reveal the latent preference and popularity semantics without making any assumption about the popularity distribution. At its core is the distillation of unbiased preference representations (i.e., user preference on item property), which are invariant to the change of popularity semantics, while filtering out the popularity feature that is unstable or outdated.
See the **quick lead-in** below.

- Q: What is popularity distribution shift in the real-world? 
![Visualization of real-world popularity distribution shifts, where both 𝑃𝑡𝑒𝑠𝑡 and 𝑃𝑡𝑒𝑠𝑡 are possible to occur but areunpredictable and unmeasurable.](Intro.pdf)
* Q: Why does the disribution shift lead to performance drops in collaborative filtering (CF) models?
+ Q: How do we solve the problem?
* Q: Why does our solution lead to robust performance?

## Installation

Main packages: PyTorch >= 1.11.0

## Run DIR

To run the code, First run the following line to install tools used in evaluation:

```bash
python setup.py build_ext --inplace
```
then run the following line to install tools used in random sampling:

```bash
python local_compile_setup.py build_ext --inplace
```

To run model on tencent_synthetic data, use main_synthetic.py, like:
```bash
python main_synthetic.py --modeltype DEBIAS_batch --infonce 1 --neg_samles -1 --n_layers 2 --dataset tencent_synthetic --need_distance 1 --lambda1 1e-2 --lambda2 1e-6 --lambda3 1e-2 
```

To run model on other datset, user main.py, like:
```bash
python main.py --modeltype DEBIAS --infonce 1 --neg_samles 64 --n_layers 2 --dataset yahoo.new --need_distance 1 --lambda1 1 --lambda2 1e-7 --lambda3 1e-1
```
Related hyperparameters are listed in the appendix of the paper.

