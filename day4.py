import torch
from torch import nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1),requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self,x:torch.tensor) ->torch.tensor:
        return self.weights * x + self.bias

x = torch.randn(3,2)



model = LinearRegression()
result = model(x)

print(result)
 