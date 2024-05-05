import torch

zero_matrix = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
zero_matrix = zero_matrix.view(5,5)
print(zero_matrix)

adjacent = torch.tensor([1,0,4,3,0,0,0,0,5,0,0,5,0,1,6,0,0,8,1,3,4,0,2,1,0])
adjacent = adjacent .view(5,5)
print(adjacent)

attention2 = zero_matrix
b = torch.topk(adjacent, k=2, dim=1, largest=True)
attention2 = attention2.scatter_(1, b.indices, b.values)
print(attention2)
