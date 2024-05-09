# HW1 - Bernoulli Naïve Bayes 
### Objective: 
To fit a Bernoilli Naïve Bayes model to the Fashion MNIST dataset. Using the model for making predictions, and generate new images from the same distribution
### Packages Used: 
NumPy, matplotlib, sklearn, os, gzip, struct, array, urlretrieve
### Approach: 
#### 1. Data Representation: 
Each image is represented by a binary vector \( \mathbf{x}(i) \in \{0,1\}^D \), where 0 and 1 represent white and black pixels respectively, and \( D=784 \). Each class label \( c(i) \) is represented by a \( K=10 \)-dimensional one-hot vector.
