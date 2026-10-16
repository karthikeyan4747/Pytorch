import torch

Tensor_A = torch.tensor([[1,2]
                         ,[2,3]
                         ,[3,4]])


Tensor_B = torch.tensor([[5,6],[6,7],[7,8]])

Tensor_C = Tensor_A * Tensor_B

print(Tensor_A.shape,Tensor_B.T.shape)

print(torch.matmul(Tensor_A,Tensor_B.T))