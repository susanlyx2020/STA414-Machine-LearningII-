# HW1 - Bernoulli Naïve Bayes 
### Objective: 
To fit a Bernoilli Naïve Bayes model to the Fashion MNIST dataset. Using the model for making predictions, and generate new images from the same distribution
### Packages Used: 
NumPy, matplotlib, sklearn, os, gzip, struct, array, urlretrieve
### Approach: 
#### 1. Data Representation: 
Each image is represented by a binary vector $\mathbf{x}(i) \in \{0,1\}^D$, where 0 and 1 represent white and black pixels respectively, and $D=784$. Each class label $c(i)$ is represented by a $K=10$-dimensional one-hot vector.
#### 2. Parameter Representation:
The parameters $\theta$ and $\pi$ are learned from the data. Due to sparsity issues with Maximum Likelihood Estimation (MLE) for $\theta$, a Maximum A Posteriori (MAP) estimator is used with a Beta prior.
#### 3. Prediction and Evaluation: 
The prediction rule is to choose the class that maximizes the log-likelihood. Accuracy is measured as the fraction of correctly predicted samples.
#### 4. Likelihood Computation: 
The average log-likelihood on the training set, as well as the training and test errors, are computed.
#### 5. Image Generation: 
Images are generated from the learned distribution using the MAP estimates. Random images are sampled from the distribution, with only 30% of the pixels observed.

# HW2 -  Loopy Belief Propagation (Loopy-BP) and MCMC in TrueSkill Model
## Loopy-BP
### Objective: 
Implementing the sum-product Loopy Belief Propagation (Loopy-BP) method to denoisis binary images.
### Packages Used: 
NumPy, matplotlib, sklearn, os, gzip, struct, array, urlretrieve
### Approach: 
#### 1. Image Representation: 
Images are represented as matrices of size $n \times n$, where each element can be either 1 or -1, with 1 representing white pixels and -1 representing black pixels.
#### 2. Noise Introduction:
Noise Introduction: Noise is introduced into the image by swapping the value of each pixel between 1 and -1 with a rate of 0.2.
#### 3.Loopy Belief Propagation (Loopy-BP): 
The Loopy-BP algorithm iteratively updates the messages of each node through a sum-product operation, computing the joint inbound message through multiplication and marginalizing the factors through summation.
#### 4. Initialization: 
Messages between neighbor pixels are initialized uniformly. Hyperparameters $J$ and $\beta$ are also initialized.
#### 5. Belief Calculation: 
Calculate the unnormalized belief for each pixel and normalize the belief across all pixels.
## MCMC in TrueSkill Model 
### Objective: 
