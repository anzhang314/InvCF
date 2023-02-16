# Invariant Collaborative Filtering to Popularity Distribution Shift
Official code of "Invariant Collaborative Filtering to Popularity Distribution Shift" (2023 WWW)

## Overview
InvCF (2022) aims to discover disentangled representations that faithfully reveal the latent preference and popularity semantics without making any assumption about the popularity distribution. At its core is the distillation of unbiased preference representations (i.e., user preference on item property), which are invariant to the change of popularity semantics, while filtering out the popularity feature that is unstable or outdated.
See the **quick lead-in** below.

- Q: What is popularity distribution shift in the real-world? 

Popularity distribution shifts are caused by the demographic, regional, and chronological diversity of human behaviors. For example, something trendy at the training stage might become very unpopular in the next period. But with popularity information injected to fit interactions in training stage, the prediction results in the next period will dramatically deviate from the user's true interest.

* Q: Why does the disribution shift lead to performance drops in collaborative filtering (CF) models?

That’s because there exists a confliction between the implicit assumption of CF models and Inevitable Popularity Distribution Shift in the real-world.  
CF models assume that the training and test data are drawn from the same distribution, but the assumption hardly holds. Popularity distribution shifts are ubiquitous and inevitable in real-world scenarios 

+ Q: How do we solve the problem?

Our solution is to filter out the unstable or outdated popularity features and learn preference representations invariant to the change of popularity semantics. We propose a new learning paradigm, called Invariant Collaborative Filtering (InvCF). We achieve the goal of generalization by implementing two principles: Disentanglement principle and Invariance principle. 

* Q: Why does our solution lead to robust performance?

In the testing scenario with popularity distribution from training, since we have fed the model with augmented popularity information along the scale, it will make a good generalization under the shift. As a result, we learn robust representation against diversified distribution changes.
 

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
python main_synthetic.py --modeltype DEBIAS_batch --infonce 1 --neg_sample -1 --n_layers 2 --dataset tencent_synthetic --need_distance 1 --lambda1 1e-2 --lambda2 1e-6 --lambda3 1e-2 
```

To run model on other datset, user main.py, like:
```bash
python main.py --modeltype DEBIAS --infonce 1 --neg_sample 64 --n_layers 2 --dataset yahoo.new --need_distance 1 --lambda1 1 --lambda2 1e-7 --lambda3 1e-1
```
Related hyperparameters are listed in the appendix of the paper.

## Reference
If you want to use our codes and datasets in your research, please cite:
```bash
@inproceedings{bc_loss,   
      author    = {An Zhang and
                   Jingnan Zheng and 
                   Xiang Wang and 
                   Yancheng	Yuan and
                   Tat-seng Chua}, 
      title     = {Invariant Collaborative Filtering to Popularity Distribution Shift},  
      booktitle = {{WWW}},  
      year      = {2023},   
}
```
