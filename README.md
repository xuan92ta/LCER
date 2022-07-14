# Learning-based Counterfactual Explanations for Top-K Recommendation

This is our Tensorflow implementation for the paper:

> Jingxuan Wen, Huafeng Liu, Liping Jing (2022). Learning-based Counterfactual Explanations for Top-K Recommendation.



## Highlights

- To characterize the effect of interacted items for generating counterfactual explanations, an optimizable importance is introduced for each interacted item. And then to further make sure the credibility of importance, the learning of importance is guided by the goal of counterfactual explanations.
- To keep the consistence between computation of importance and generation of counterfactual explanations, counterfactual explanations are learned with the learning of importance in an end-to-end manner, which also avoids the sub-optimality of counterfactual explanations brought by search strategy.
- To evaluate the effectiveness, a series of experiments are conducted on four public datasets, and the experimental results demonstrate signiﬁcant improvements of LCER, both quantitatively and qualitatively.



## Environment Requirement

The code has been tested running under Python 3.7.13. The required packages are as follows:

- tensorflow == 1.14.0
- numpy == 1.21.2
- scipy == 1.6.2
- tqdm == 4.63.0
- bottleneck == 1.3.4



## Example to Run the Codes

1. Download the dataset: [ML-100K/1M](https://grouplens.org/datasets/movielens/), [Alishop](https://jianxinma.github.io/disentangle-recsys.html), or [Epinions](http://www.trustlet.org/downloaded_epinions.html).

2. Train the recommendation model Mult-VAE in `VAE_ML20M_WWW2018.ipynb`. The code of Mult-VAE is published in https://github.com/dawenl/vae_cf.

   > Liang D, Krishnan R G, Hoffman M D, et al. Variational autoencoders for collaborative filtering[C]//Proceedings of the 2018 world wide web conference. 2018: 689-698.

3. Generate counterfactual explanations.

   The parameters have been clearly introduced in `parse.py`. 

   - ML 100K dataset

     ```
     python LCER.py --dataset=ml-100k --lam=0.01 --alpha=0.1
     ```

   - ML 1M dataset

     ```
     python LCER.py --dataset=ml-1m --lam=0.02 --alpha=0.04
     ```

   - Alishop dataset

     ```
     python LCER.py --dataset=alishop --lam=0.03 --alpha=0.08
     ```

   - Epinions dataset

     ```
     python LCER.py --dataset=epinions --lam=0.01 --alpha=0.1
     ```

     

