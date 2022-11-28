
# A Few-shot Learning Method for Spatial Regression
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang, Rui Guob

 This code implemented by Li-Li Bao for  A Few-shot Learning Method for Spatial Regressionis is based on 
 
  **PyTorch 1.11.0**
  
  **python 3.8**. 
  
  **GPU is NVIDIA GeForce RTX 3080**.
## The structure of our SMANP:
  We propose a few-shot learning method for spatial regression tasks. Our method combines the advantages of NN and stochastic process to learn the spatial relationship among attributes come from few samples by efficiently using backpropagation to optimize the expected prediction performance, predict the target value and give the prediction uncertainty . Specifically, we learn on the one hand a wide range of distribution family for data based on stochastic processes
whose parameters (mean and covariance functions of Gaussian processes) are parameterized by NNs. On the other hand, we capture the fine-grained information between attributes and the spatial correlation between objects based on different attention mechanisms, so as to obtain good predictions
of target attributes and quantify the uncertainty under the framework of meta-learning. We refer to our model as Spatial Multi-Attentional Neural Process (SMANP).
![structure](https://user-images.githubusercontent.com/92556725/204268754-f5857a26-8abf-4063-a5db-f785351e562d.jpg)
Model Architecture of SMANP. Encoders and decoders make up SMANP. Latent path and deterministic path are two of the key components of the encoder.
## Dataset
  Since our real data is confidential, We uploaded the modified reservoir thickness prediction dataset to help you to run  the code. This leads to the difference between your running results and the description in the manuscript.
  The data set is data.csv,which includes 459 logging data, and each logging is given an ID number. you can  run split.py to  split data.csv into the training set and test set according to different proportions to verify the effect of the network. They include 14 auxiliary variables such as Line, CMP, Freq, and so on. Please refer to the manuscript for the actual significance of auxiliary variables. The reservoir thickness value is our prediction variable. SValue in training set and test set is known.
## Implementation of SMANP
  You can run train_smanp.py to relize our SMANP.
