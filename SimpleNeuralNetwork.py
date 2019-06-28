
# coding: utf-8

# # Creating a simple Neural Network Using python
# 
# https://becominghuman.ai/making-a-simple-neural-network-2ea1de81ec20
# 
# ![image.png](attachment:image.png)
# 
# This Neural Network simply receives an input then modify said input by a weight, and finally return an output based on the addition of a layer.
# 

# In[1]:


import numpy as np

# -------------------------------------------------------------------------
# Declare variables
# -------------------------------------------------------------------------

inputs = np.array([0,1,0,0])
weights = np.array([0.00,0.00,0.00,0.00])

expectedResult = 1

learningRate = 0.2

trials = 6


# In[2]:


# -------------------------------------------------------------------------
# Define the network evaluation function which returns the result
# -------------------------------------------------------------------------

def eval_Network(inputVector, weightVector):
    
    networkResult = 0

    for i in range(len(inputVector)):
        layerValue = inputVector[i] * weightVector[i]
        networkResult += layerValue
    
    return networkResult


# -------------------------------------------------------------------------
# Define the Error function
#      Error = Desired Output - Neural Net Output
# -------------------------------------------------------------------------

def eval_NetworkError(desired,actual):
    
    return (desired - actual)


# -------------------------------------------------------------------------
# Define the NN learning function
# -------------------------------------------------------------------------

def learn(inputVector, weightVector):
    for i in range(len(weightVector)):
        if inputVector[i] > 0:
            weights[i] = (weightVector[i] + learningRate)
    
    return weights

# -------------------------------------------------------------------------
# Define function for repetition (epochs)
# -------------------------------------------------------------------------

def train(epoch):
    
    print("----------+---------------+--------------+-----------")
    print("Epoch     | NN Output     | Accuracy     | Error     ")
    print("----------+---------------+--------------+-----------")
    
    for i in range(epoch):
        
        output_NN = eval_Network(inputs,weights)
        
        learn(inputs,weights)
        
        NN_Error = eval_NetworkError(expectedResult,output_NN)

        print(i+1,"/",trials, "     ", 
              round(output_NN,2), "           ", 
              round((1-NN_Error)*100,2), "%       ", 
              round(NN_Error,2))


# In[3]:


# -------------------------------------------------------------------------
#Run the Model to 100% accuracy at trial 6
# -------------------------------------------------------------------------

train(trials)

