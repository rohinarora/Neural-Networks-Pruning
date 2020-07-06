### Requirements
Main dependency is pytorch. Based on python 3.7

```conda env create --name envname --file=environments.yml```

### Data

```source create_dataset.sh```

Creates test and train DIR. Uses 1000 images for training and 400 for test. Feel free to change this by updating the script

### Code

### Key points

*  Based on the paper [Pruning Convolutional Neural Networks for Resource Efficient Inference
](https://arxiv.org/abs/1611.06440)
* Recurring theme
  * Focus on transfer learning
  * Greedy criteria-based pruning of feature maps from convolutional layers
* Many pruning techniques require per layer sensitivity analysis which adds extra computations. For example, as descibed in - [ Pruning Filters For Efficient Convnets](https://github.com/rohinarora/Neural-Networks-Pruning/tree/master/1.%20Pruning%20Filters%20For%20Efficient%20Convnets)
  * In contrast, this approach relies on global rescaling of criteria for all layers and does not require sensitivity estimation.
* Global optima of pruning combinatorially hard/exponential/infeasible. Use greedy methods.
* Algorithm-> (success hinges on employing the right pruning criterion)
  * Fine-tune the network until convergence on the target task
  * Alternate iterations of pruning and further fine-tuning. Feature maps are ranked globally, and then pruned
  * Stop pruning after reaching the target trade-off between accuracy and pruning objective
* Taylor pruning criterion a good heuristic.
* Greedy criteria based on Taylor expansion that approximates the change in the cost function induced by pruning network parameters.
* This repo is inspired by Jacob's nice [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
