# Cmpe491-sadiuysal
CMPE 491 Project Repository

## Updates

### 26.10.2020-01.11.2020
Completed 2 chapter in hands-on tutorial. 

[Chapter 1 - Introduction to adversarial robustness](https://adversarial-ml-tutorial.org/introduction/)
#### Chapter Parts:
##### Diving right in:
Done basic image classification in PyTorch.
##### Some introductory notation:
Defining the model, or hypothesis function.
Defining a loss function.
##### Creating an adversarial example:
So how do we manipulate this image to make the classifier believe it is something else?
Created successful adversarial example in Pytorch. Visualize added perturbation. 
##### Targeted attacks:
This is known as a “targeted attack”, and the only difference is that instead of trying 
to just maximize the loss of the correct class, we maximize the loss of the correct class
while also minimizing the loss of the target class.
Created succesful targeted adversarial attack.
##### A brief (incomplete) history of adversarial robustness:
* Origins (robust optimization)
* Support vector machines
* Adversarial classification (e.g. Domingos 2004)
* Distinctions between different types of robustness (test test, train time, etc)
* Many proposed defense methods
* Many proposed attack methods
* Exact verification methods
* Convex upper bound methods
##### Adversarial robustness and training:
As an alternative to the traditional risk, we can also consider an adversarial risk.
This is like the traditional risk, except that instead of suffering the loss on each 
sample point, we suffer the worst case loss in some region around the sample point.

But how do we compute the gradient of the inner term now, given that the inner function
itself contains a maximization problem? The answer is fortunately quite simple in practice,
and given by [Danskin’s theorem](https://en.wikipedia.org/wiki/Danskin%27s_theorem). For the purposes of our discussion, it states that the 
gradient of the inner function involving the maximization term is simply given by the 
gradient of the function evaluated at this maximum. 



