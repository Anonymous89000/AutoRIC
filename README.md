# AutoRIC

AutoRIC is used to solve constraint-based neural network repair problems. In this project, neural network repair tasks related to fairness and robustness were carried out.

```
census_FFNN_Fairness:
```

Fairness optimization experiment of FFNN based on census dataset.

```
Jigsaw_RNN_Fairness:
```

Expand the types of networks that can be optimized and perform fairness optimization tasks on RNNs.

```
MNIST_CNN_Robustness:
```

Expand the types of validation properties that can be optimized and optimize the robustness of neural networks on the MNIST dataset.

| Dataset |  NN  |  Property  |
| :-----: | :--: | :--------: |
| census  | FFNN |  Fairness  |
| Jigsaw  | RNN  |  Fairness  |
|  MNIST  | CNN  | Robustness |



## census_FFNN_Fairness

### Quick Start

Run the following three scripts directly to perform a complete neural network repair (for FFNN on census).

```
quadratic_fit.py
```

Run the script to perform a quadratic fit and save the quadratic parameters.(Inside the script, the number of neurons to be repaired and the amount of data used for fitting can be modified)

```
constraint_by_linearize.py
```

Run the script to generate constraints through network linearization.

```
total23.py
```

Run the script to convex the quadratic and constraints, and then test and output the optimization results (using the quadratic parameters and linear constraints from the first two steps).

### File Description

```
constraint_by_linearize.py
```

The linear constraints that need to be satisfied for optimization are obtained through linearization and saved in the constraint folder in the form of an inequality group.

```
quadratic_fit.py
```

The quadratic fit obtains the quadratic correlation coefficient and is saved in quadratic_para.

```
cvx_optimize.py
```

The quadratic programming is solved according to the fitted quadratic form and the constraints constructed by linearization to obtain the values of the parameters to be optimized.

```
total23.py
```

The overall program that combines the optimization processes can be directly run and the optimization results can be observed and printed out. The module can select the parameters to be optimized and the constraints used in the quadratic programming. Selecting different input parameters will produce different optimization results.

```
dichotomy_optimize.py
```

The dichotomy method is used to adjust the network optimization results, which is not highlighted in the paper.

```
linear_optimize.py
```

Make a linear optimization of the network and form a control test with convex optimization.

```
retrainCalFairness.py
```

Testing the method of re -training on optimizing the fairness of neural networks.

## Jigsaw_RNN_Fairness

```
main.py
```

Run the main function to complete the fairness repair of RNN.

## MNIST_CNN_Robustness

```
main.py
```

Run the main function to complete the robustness repair of CNN.

