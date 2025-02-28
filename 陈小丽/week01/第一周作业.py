import numpy as np
a = [1,2,3,4,5,6]
b = np.array(a)

c = np.array([1,2,3,4,5,6], int)

d = np.array([[1,2,3],[4,5,6],[7,8,9]])

e = np.zeros((2,3), dtype = float)
f = np.zeros((2,3), dtype = int)

g = np.ones((2, 2)

h = np.arange(1, 5, 0.25)

i = np.eye(4)

np.random.random(7)

a,b = 0,0.2
np.random.normal(a,b,7)

j = np.array([(1,2), (3,4), (5,6)])
j[0]
j[1:]
j[: , :1]
j[1][1]

k = np.array([1,2,3])
for i in k:
print(i)

l = np.array([(1,2), (3,4), (5,6)])
for i,j in l:
print(i/j)

m = np.array([(1,2,3,4), (5,6,7,8), (9,10,11,12)])
print("ndim:", m.ndim)
print("shape:", m.shape)
print("size", m.size)
print("dtype", m.dtype)

o = np.array([(1,2), (3,4)])
print(3 in o)
print(5 in o)

p = np.zeros([2,3,4])

q.reshape(24)

r = np.array([(1,2,3,4), (5,6,7,8), (9,10,11,12)])
r.transpose()
r.T

s = np.array([(1,2,3), (4,5,6), (7,8,9)])
s.flatten()

a = np.array([1,2,3]) 
a.shape

a = a[ :, np.newaxis]
a.shape

a = np.ones((3,3))
b = np.array([(-1,1,2),(2,-1,1)])
print(a)
print(b)
a + b
a - b
a * b
a / b
a.sum()
a.prod()
a.mean()
a.var()
a.std()
a.max()
a.min()

a = np.array([100.3, 38, 5.6])
print("argmax:", a.argmax())
print("argmin:", a.argmin())
print("ceil:", np.ceil(a))
print("floor:", np.floor(a))
print("rint:", np.rint(a))
a.sort()

inmport numpy as np
m1 = np.array([[1, 2], [3, 4] , dtype=np.float64)
m2 = np.array([[5, 6], [7, 8] , dtype=np.float64)
result_dot = np.dot(m1, m2)
result_at = m1 @ m2
print(m1)
print(m2)
print(result_dot)
print(result_at)

manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)
for i in range(m1.shape[0]):
 for j in range(m2.shape[1]):
 manual_result[i, j] = 0
 for k in range(m1.shape[1]):
 print(f"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}")
 manual_result[i, j] += m1[i, k] * m2[k, j]
 print(f"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\n")
print(manual_result)


a = np.array([1,2,3])
b = np.array([4,5,6])
a + b
a - b

a = np.array([(1,2), (2,2), (3,3), (4,4)]) 
b = np.array([-1,1]) 
a + b
a - b




data = [[1,2],[3,4]]
x_data=torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) 
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) 
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

m = torch.ones(5,3, dtype=torch.double)
n = torch.rand_like(m, dtype=torch.float)
print(m.size()) # torch.Size([5,3])
torch.rand(5,3)
torch.randn(5,3)
torch.normal(mean=.0,std=1.0,size=(5,3)
torch.linspace(start=1,end=10,steps=20)

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


if torch.cuda.is_available():
 tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(tensor, "\n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t:{t}")
print(f"n:{n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

import torch
from torchviz import make.dot
A = torch.randn(10, 10,requires_grad=True)
b = torch.randn(10,requires_grad=True)
c = torch.randn(1,requires_grad=True)
x = torch.randn(10, requires_grad=True)
result = torch.matmul(A, x.T) + torch.matmul(b, x) + c
dot = make_dot(result, params={'A': A, 'b': b, 'c': c, 'x': x})
dot.render('expression', format='png', cleanup=True, view=False)
