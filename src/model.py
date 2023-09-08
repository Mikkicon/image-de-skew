import torch
import torch.nn as nn
import torch.nn.functional as F

class SkewNet(nn.Module):
    # Define the neural network that has some learnable parameters (or weights)
    # Iterate over a dataset of inputs
    # Process input through the network
    # Compute the loss (how far is the output from being correct)
    # Propagate gradients back into the network’s parameters
    # Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

  def __init__(self, *args, **kwargs) -> None:
    super(SkewNet, self).__init__(*args, **kwargs)

    self.conv1 = nn.Conv2d(1, 6, 3) # TODO what are channels

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # TODO what?
    
    x = torch.flatten(x, 1) # TODO what?

    return x
  
  # TODO Loss function
  
def main():
  net = SkewNet()
  print(net)

if __name__ == '__main__':
  main()