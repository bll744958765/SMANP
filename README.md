
# SDANP: A Dual-attention-based Method for Spatial Small Sample Prediction
LiLi Bao,  ChunXia Zhang, JiangShe Zhang, Rui Guo, ChengLi Tan, Kai Sun

 This code implemented by Li-Li Bao for  A Few-shot Learning Method for Spatial Regressionis is based on 
 
  **PyTorch 1.11.0**
  
  **python 3.8**. 
  
  **GPU is NVIDIA GeForce RTX 3080**.
## The structure of our SDANP:
  Given spatial data samples with auxiliary explanatory variables and target response variables observed at a small set of spatial locations, the spatial prediction tasks aim to learn a model with this labeled dataset to predict the target variable when the values of the spatial and auxiliary variables are given.
There are many deep learning methods that have achieved impressive success in spatial small sample prediction tasks. 
However, these methods focus more on extracting information from auxiliary variables and ignore the spatial dependencies contained in spatial data, which may result in suboptimal performance for spatial prediction tasks. 
To alleviate this problem, in this paper we propose a Spatial Dual-branch Attention Neural Process (SDANP) model, which decouples the input variables into spatial and auxiliary variables and uses two parallel modules to extract the target-related information from the spatial and auxiliary variables, respectively.
Specifically, SDANP uses an encoder-decoder architecture similar to that of Neural Process (NP) to predict the distribution of targets. 
In the encoder, a Laplace attention module is used to focus on the spatial dependencies from the spatial variables and a multi-head attention module is used in parallel to fully extract the correlations from the auxiliary variables. 
The decoder aggregates the encoder outputs with the spatial and auxiliary variables of the target point to be predicted, yielding the posterior distribution of the predicted target.  
This posterior distribution not only provides a good prediction of the target but also quantifies the uncertainty associated with the prediction.
Furthermore, the modular design of SDANP allows for flexible adaptation to data forms with only spatial or auxiliary variables, or both, without requiring significant model modifications. 
Experimental results on synthetic and real datasets demonstrate that the proposed SDANP method achieves state-of-the-art results in terms of both predictive performance and reliability.

![structure](https://github.com/bll744958765/SMANP/assets/92556725/f441bb58-1e46-4b11-b3a7-39e8e5938440)

Model Architecture of SDANP. Encoders and decoders make up SDANP. Latent path and deterministic path are two of the key components of the encoder.
## Dataset and Experiment
We tested the effect of SDANP on synthetic and real spatial. 

### Simulation studies
We synthesised datasets with spatial correlation relationships and then make forecasts based on them. 
We demonstrate the SDANP’s effectiveness and robustness on synthesised datasets with various numbers of training examples. In the beginning, we divide a synthesised dataset into a training and validation set, with training being performed on the training set and validation being performed on the validation dataset. Then, we use the finest trained weights to test on the new synthesised dataset, i.e., the test set.


## Implementation of SDANP on simulation dataset
You can run split.py to load train_simulation and split it into training and validation sets for SMANP to learn. test_simulation.CSV is a new simulation data set that can be used to test the performance of SDANP。 The performance of SDANP on the test_simulation.csv is shown in Table1 and Figure 2.

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
                     	 <td>0.1883</td>
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
                       <td>1.0569</td>  
                       <td>1.3155</td> 
                     	 <td>0.8330</td> 
                       <td>0.9115</td> 
                     	 <td>0.1889</td>
                   </tr>
                
               </table>

![simulation](https://user-images.githubusercontent.com/92556725/226334344-f9df5dcc-d096-47e3-893b-49fe7342553f.png)

