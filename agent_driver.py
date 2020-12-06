#See devices
import os        
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#Set order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
#Select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from agent import Agent


config = {
    'batchSize': 1024,
    'nInputs': 128, 
    'nOutputs': 2, 
    'learningRate':0.01, 
    'dqnConfig':
          {
              'maxMemory': 50000,
              'discount': 0.9,
              'actionMap': {
                  2: 0,
                  3: 1,
                  0: 2,
                  1: 3
                  }
          },
    'trainingEpochs': 10,
    'epsilon': 1.0,
    'epsilonDecayRate': 0.95
}
nsessions = 10000   #Number of episodes or sessions to be played and trained on in total
agent = Agent(nsessions, config)
agent.train()