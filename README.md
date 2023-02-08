# Invariant Collaborative Filtering to Popularity Distribution Shift
Official code of "Invariant Collaborative Filtering to Popularity Distribution Shift" (2023 WWW)

## Overview
InvCF (2022) aims to discover disentangled representations that faithfully reveal the latent preference and popularity semantics without making any assumption about the popularity distribution. At its core is the distillation of unbiased preference representations (i.e., user preference on item property), which are invariant to the change of popularity semantics, while filtering out the popularity feature that is unstable or outdated.
See the **quick lead-in** below.

- Q: What is popularity distribution shift in the real-world? 
![Visualization of real-world popularity distribution shifts, where both 𝑃𝑡𝑒𝑠𝑡 and 𝑃𝑡𝑒𝑠𝑡 are possible to occur but areunpredictable and unmeasurable.](https://github.com/anzhang314/InvCF/blob/main/Intro.pdf)
* Q: Why does the disribution shift lead to performance drops in collaborative filtering (CF) models?
+ Q: How do we solve the problem?
* Q: Why does our solution lead to robust performance?
