import torch

tensor = torch.tensor([[[[1,2,3],[1,2,3],[1,2,3]]]])

print(tensor.shape)
print(tensor.ndim)


Random_Tensor = torch.rand(1,1,3,3)
print(Random_Tensor)


range=torch.arange(0,10)
print(range)

like_zeroes = torch.zeros_like(input=Random_Tensor)
print(like_zeroes)