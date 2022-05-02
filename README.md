# Alzheimer's classification

# Aim of the project
The purpose is to develop testing algorithms for Alzheimer's Disease classification. I am using a Convolutional neural network to classify the patient's status. The input data is an MRI (Magnetic resonance imaging) of the brain. The number of output classes is 4. They are non-demented, very mild demented, mild demented and moderate demented. Check out the structure of the Convolutional Neural Network.
|Layer (type)| Output Shape      | Param #|
|------------|-------------------|--------|
|Conv2d-1    |[-1, 3, 124, 124]  |78      |
|MaxPool2d-2 |[-1, 3, 62, 62]    |0       |
|Conv2d-3    |[-1, 6, 58, 58]    |456     |
|MaxPool2d-4 |[-1, 6, 29, 29]    |0       |
|Linear-5    |[-1, 120]          |605,640 |
|Linear-6    |[-1, 84]           |10,164  |
|Linear-7    |[-1, 4]            |340     |

Total params: 616,678 <br/>
Trainable params: 616,678 <br/>
Non-trainable params: 0 <br/>
Input size (MB): 0.06 <br/>
Forward/backward pass size (MB): 0.63 <br/>
Params size (MB): 2.35 <br/>
Estimated Total Size (MB): 3.05 <br/>

# Used data
I used the open-source dataset provided in Kaggle(https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). I converted the data into CSV format to be easier for training in PyTorch. I split the data into 80% of every class for training and 20% of every class for testing.

# Training the model
I used Cross Entropy Loss Function and Adam optimizer
# Results:
|Accuracy of:      |%                |
|------------------|-----------------|
|Network           |97.890625        |
|Non demented      |98.59375         |
|Very mild demented|97.09821428571429|
|Mild demented     |97.20670391061452|
|Moderate demented |100.0            |

**Graph of the loss function for 100 epochs**<br/>
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/loss.png?raw=true)<br/>
**Graph of the test accuracy for 100 epochs**<br/>
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/acc.png?raw=true)<br/>
**Graph of the test and trainign accuracy for 100 epochs**<br/>
![alt text](https://github.com/delyanbg05/AlzhimerClassification/blob/master/results/acc_cmp.png?raw=true)
