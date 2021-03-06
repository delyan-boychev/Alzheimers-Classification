# Alzheimer's classification

## Aim of the project
The purpose is to develop testing algorithms for Alzheimer's Disease classification. I am using a Convolutional neural network to classify the patient's status. The input data is an MRI (Magnetic resonance imaging) of the brain. The number of output classes is 4. The classes are non-demented, very mild demented, mild demented and moderate demented. Check out the structure of the Convolutional Neural Network.
|Layer (type)|Input shape  | Output Shape      |
|------------|-------------|-------------------|
|Conv2d-1    |(1, 128, 128)|(3, 124, 124)      |
|MaxPool2d-2 |(3, 124, 124)|(3, 62, 62)        |
|Conv2d-3    |(3, 62, 62)  |(6, 58, 58)        |
|MaxPool2d-4 |(6, 58, 58)  |(6, 29, 29)        |
|Linear-5    |(5046)       |(120)              |
|Linear-6    |(120)        |(84)               |
|Linear-7    |(84)         |(4)                |

Total params: 616,678 <br/>
Trainable params: 616,678 <br/>
Non-trainable params: 0 <br/>
Input size (MB): 0.06 <br/>
Forward/backward pass size (MB): 0.63 <br/>
Params size (MB): 2.35 <br/>
Estimated Total Size (MB): 3.05 <br/>

## Used data
I used the open-source dataset provided in Kaggle(https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). I created a CSV file with annotations to be easier to train with PyTorch. I split the data into 80% of every class for training and 20% of every class for testing.

## Training the model
I used Cross Entropy Loss Function and Adam optimizer
## Results:
|Accuracy of:      |%                |
|------------------|-----------------|
|Network           |97.890625        |
|Non demented      |98.59375         |
|Very mild demented|97.09821428571429|
|Mild demented     |97.20670391061452|
|Moderate demented |100.0            |


### Graph of train loss for 100 epochs training
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/loss_train.png?raw=true)<br/>

### Graph of train and test loss for 100 epochs training
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/loss_cmp.png?raw=true)<br/>
### Graph of test set accuracy for 100 epochs training
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/acc.png?raw=true)<br/>
### Graph of test and training set accuracy for 100 epochs training
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/acc_cmp.png?raw=true)
