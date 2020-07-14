### Requirements
Main dependencies are pytorch and cuda. Based on python 3.7

```conda env create --name envname --file=environments.yml```
* Tested with CUDA 10.2

### Data

```source create_dataset.sh```

* Creates test and train DIR. Uses 1000 images for training and 400 for test. Feel free to change this by updating the script
* Requires kaggle CLI API. Alternatively download manually from https://www.kaggle.com/c/dogs-vs-cats/data and use then use the above script (commenting the kaggle API call)

### Code
* Refer vgg16_dogs_vs_cats_dataset.ipynb
* Transfer learning on pre trained VGG, replacing the classifier head for dogs_vs_cats dataset (2 class label)
```
model = models.vgg16(pretrained=True)
self.features = model.features #use the pre-trained feature head
for param in self.features.parameters(): #freeze the feature head
    param.requires_grad = False
self.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(25088, 4096), #output of last conv is 7x7x512 (25088). Feed that to FC layer
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 2)) #final classifer is 2 category
def forward(self, x):
    x = self.features(x) #x.shape torch.Size([32, 512, 7, 7]) (batchsize 32)
    x = x.view(x.size(0), -1) #x.shape torch.Size([32, 25088])
    x = self.classifier(x) #x.shape torch.Size([32, 2])
  return x
```
* Initially fine tune the VGG on dogs_vs_cats dataset for 50 epochs, getting ___ accuracy
```
initial_training_obj.train(epoches=50)
```
* Prune 70% of the network parameters
  * Prune 32 filters in one iteration, re-train for 10 epochs and prune again till 70% of the network parameters are pruned
```
pruner_obj.prune(percentage_to_prune=70, num_filters_to_prune_per_iteration=32)
```


### Results

### Key points
*  Based on the paper [Pruning Convolutional Neural Networks for Resource Efficient Inference
](https://arxiv.org/abs/1611.06440)
* Refer [this](https://github.com/rohinarora/Deep-Learning-Papers/blob/master/main/pcnnfrefi.md) for key points
* Inspired by Jacob's nice [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
