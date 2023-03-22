
# A Few-shot Learning Method for Spatial Regression
Li-Li Bao,  Chun-Xia Zhang, Jiang-She Zhang, Rui Guo， Cheng-Li Tan

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
## Dataset and Experiment
We tested the effect of SMANP on one simulation dataset and two actual seismic data sets. Two actual seismic data sets are provided by The Bureau of Geophysical Prospecting Inc., China National Petroleum Corporation. The first piece of work falls under the multi attribute fusion reservoir thickness prediction task, while the second is the horizontal velocity prediction task

### Simulation studies
We simulate datasets with spatial correlation relationships and then make forecasts based on them. 
We demonstrate the SMANP’s effectiveness and robustness on simulated datasets with various numbers of training examples. In the beginning, we divide a simulation dataset into a training and validation set, with training being performed on the training set and validation being performed on the validation dataset. Then, we use the finest trained weights to test on the new simulation dataset, i.e., the test set.


### Reservoir Thickness 
  Since our real data is confidential, We uploaded the modified reservoir thickness prediction dataset  to help you to run  the code. This leads to the difference between your running results and the description in the manuscript.
  The data set is data_reservoir, which includes 459 logging data, and each logging is given an ID number. They include 14 auxiliary variables such as Line, CMP, Freq, and so on. Please refer to the manuscript for the actual significance of auxiliary variables. The reservoir thickness value is our prediction variable. SValue in training set and test set is known.  you can  run split.py to  split data.csv into the training set and test set according to different proportions to verify the effect of the network.
## Implementation of SMANP on simulation dataset
You can run split.py to load train_simulation and split it into training and validation sets for SMANP to learn. test_simulation.CSV is a new simulation data set that can be used to test the performance of SMANP。 The performance of SMANP on the test_simulation.csv is shown in Table1 and Figure 2.

<table>
    <tr>
        <td>Dataset</td> 
        <td>Ratio</td> 
        <td>MAE</td> 
        <td>RMSE</td> 
        <td>R^2</td> 
        <td>CCC</td> 
        <td>Var</td> 
   </tr>
   <tr>
        <td rowspan="3">Valid dataset</td>    
        <td>0.1</td> 
        <td>1.1691</td> 
        <td>1.4809</td> 
        <td>0.7937</td> 
       	<td>0.8859</td> 
        <td>0.1856</td> 
    </tr>
    <tr>
        <td>0.3</td> 
        <td>1.0562</td>  
        <td>1.3351</td> 
      	 <td>0.8346</td>
        <td>0.9126</td> 
        <td>0.1614</td> 
    </tr>
    <tr>
        <td>0.5</td> 
        <td>1.0477</td>  
        <td>1.3048</td> 
      	 <td>0.8412</td> 
        <td>0.9161</td> 
      	 <td>0.1523</td>
    </tr>
    <tr>
       <td rowspan="3">Test dataset</td>    
  		   <td>0.1</td> 
      	<td>1.1584</td> 
       <td>1.4629</td> 
       <td>0.7908</td> 
       <td>0.8833</td> 
       <td>0.1858</td> 
    </tr>
    <tr>
        <td>0.3</td> 
        <td>1.0607</td>  
        <td>1.3395</td> 
      	 <td>0.8252</td> 
        <td>0.9062</td> 
      	 <td>0.1624</td>
    </tr>
    <tr>
        <td>0.5</td> 
        <td>1.0669</td>  
        <td>1.3155</td> 
      	 <td>0.8330</td> 
        <td>0.9115</td> 
      	 <td>0.1589</td>
    </tr>
 
</table>

![simulation](https://user-images.githubusercontent.com/92556725/226334344-f9df5dcc-d096-47e3-893b-49fe7342553f.png)

