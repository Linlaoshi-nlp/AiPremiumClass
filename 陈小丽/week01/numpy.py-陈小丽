{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([6., 7., 8., 9., 10., 11.])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# 创建ndarray数组\n",
    "arr = np.array([6,7,8,9], int)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[2, 3],\n",
       "       [4, 5],\n",
       "       [8, 9]])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([(1,2), (4,5), (7,8)])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0.],\n",
       "       [0., 0., 0.],\n",
       "       [0., 0., 0.]], dtype=float64)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a1 = np.zeros((3,3), dtype=np.float32)\n",
    "a1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,\n",
       "       7.5, 8. , 8.5, 9. , 9.5])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a2 = np.arange(1,10, 0.5)\n",
    "a2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0., 0., 0.],\n",
       "       [0., 1., 0., 0., 0.],\n",
       "       [0., 0., 1., 0., 0.],\n",
       "       [0., 0., 0., 1., 0.],\n",
       "       [0., 0., 0., 0., 1.]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a3 = np.eye(5)\n",
    "a3  # one-hot编码"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.71785019, 0.18417888, 0.94725575, 0.01480714, 0.61884446,\n",
       "       0.94935988])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a4 = np.random.random(5)  # 模型运算参数初始值\n",
    "a4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.11784015, -0.1895922 , -2.28290899, -2.84400546,  1.20310538])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a5 = np.random.normal(-1, 1.1, 5)  # 模型运算参数初始值\n",
    "a5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 4 6]\n"
     ]
    }
   ],
   "source": [
    "a6 = np.arange(1,7).reshape(3,2)\n",
    "# print(a6)\n",
    "print(a6[:,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 2 3\n",
      "3 4 5\n",
      "4 5 6\n"
     ]
    }
   ],
   "source": [
    "a7 = np.array([(1,2,3), (3,4,5), (4,5,6)])\n",
    "\n",
    "for i,j,k in a7:\n",
    "    print(i,j,k)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ndim: 2\n",
      "shape: (3, 3)\n",
      "size 9\n",
      "dtype int64\n"
     ]
    }
   ],
   "source": [
    "a = np.array([(1,2,3), (4,5,6), (7,8,9)])\n",
    "print(\"ndim:\", a.ndim)\n",
    "print(\"shape:\", a.shape)\n",
    "print(\"size\", a.size)\n",
    "print(\"dtype\", a.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "True\n"
     ]
    }
   ],
   "source": [
    "print(3 in a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1  2  3  4  5  6  7  8  9 10 11 12]\n",
      "(12,)\n",
      "[[[ 1]\n",
      "  [ 2]\n",
      "  [ 3]\n",
      "  [ 4]]\n",
      "\n",
      " [[ 5]\n",
      "  [ 6]\n",
      "  [ 7]\n",
      "  [ 8]]\n",
      "\n",
      " [[ 9]\n",
      "  [10]\n",
      "  [11]\n",
      "  [12]]]\n",
      "(3, 4, 1)\n"
     ]
    }
   ],
   "source": [
    "a7 = np.arange(1,13)\n",
    "print(a7)\n",
    "print(a7.shape)\n",
    "\n",
    "a7 = a7.reshape(3,4,1)  # 维度大小乘积 == 元素个数\n",
    "print(a7)\n",
    "\n",
    "print(a7.shape)  # 高维矩阵，每个维度都有含义\n",
    "\n",
    "# 加载图像数据\n",
    "# img.shape [1,3,120,120] 1个样本，3颜色特征通道，120高，120宽"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[2 3]\n",
      " [4 5]\n",
      " [8 9]]\n",
      "[[2 4 8]\n",
      " [3 5 9]]\n"
     ]
    }
   ],
   "source": [
    "print(a)\n",
    "a = a.T\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 4 8 3 5 9]\n"
     ]
    }
   ],
   "source": [
    "a = a.flatten()\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 1, 2)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a8 = np.array([(1,2), (3,4), (5,6)])\n",
    "a8 = a8[:,np.newaxis,:]  # [3,1,2]\n",
    "a8.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 1.]\n",
      " [1. 1.]]\n",
      "[[-1  1]\n",
      " [-1  1]]\n",
      "\n",
      "[[0. 2.]\n",
      " [0. 2.]]\n",
      "\n",
      "[[2. 0.]\n",
      " [2. 0.]]\n"
     ]
    }
   ],
   "source": [
    "a = np.ones((2,2))\n",
    "b = np.array([(-1,1),(-1,1)])\n",
    "print(a)\n",
    "print(b)\n",
    "print()\n",
    "print(a+b)\n",
    "print()\n",
    "print(a-b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.prod()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mean: 5.0\n",
      "var: 2.6666666666666665\n",
      "std: 1.632993161855452\n"
     ]
    }
   ],
   "source": [
    "a = np.array([5,3,7])\n",
    "print(\"mean:\",a.mean())\n",
    "print(\"var:\", a.var())\n",
    "print(\"std:\", a.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "argmax: 2\n",
      "argmin: 0\n",
      "ceil: [11. 14. 15.]\n",
      "floor: [10. 13. 14.]\n",
      "rint: [10. 14. 15.]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([10.02, 13.8, 14.9])\n",
    "print(\"argmax:\", a.argmax()) # 最大值的索引\n",
    "print(\"argmin:\", a.argmin()) # 最小值的索引\n",
    "\n",
    "print(\"ceil:\", np.ceil(a))  # 向上取整\n",
    "print(\"floor:\", np.floor(a)) # 向下取整\n",
    "print(\"rint:\", np.rint(a))  # 四舍五入"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.08683621, 0.258836  , 0.28688792, 0.3096032 , 0.35518514,\n",
       "       0.43619354, 0.58171037, 0.89288993, 0.89982167, 0.95796287])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.random.rand(10)\n",
    "a.sort()  # 排序\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 4, 10, 18],\n",
       "       [28, 40, 54]])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a9 = np.array([[1,2,3],[4,5,6]])\n",
    "b9 = np.array([[4,5,6],[7,8,9]])\n",
    "\n",
    "a9 * b9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "矩阵 1:\n",
      "[[1. 1.]\n",
      " [2. 2.]]\n",
      "矩阵 2:\n",
      "[[5. 6.]\n",
      " [7. 8.]]\n",
      "使用 np.dot 得到的矩阵乘法结果:\n",
      "[[12. 14.]\n",
      " [24. 28.]]\n",
      "使用 @ 运算符得到的矩阵乘法结果:\n",
      "[[12. 14.]\n",
      " [24. 28.]]\n",
      "1.0 * 5.0 = 5.0\n",
      "1.0 * 7.0 = 7.0\n",
      "结果矩阵[1,1]:12.0\n",
      "\n",
      "1.0 * 6.0 = 6.0\n",
      "1.0 * 8.0 = 8.0\n",
      "结果矩阵[1,2]:14.0\n",
      "\n",
      "2.0 * 5.0 = 10.0\n",
      "2.0 * 7.0 = 14.0\n",
      "结果矩阵[2,1]:24.0\n",
      "\n",
      "2.0 * 6.0 = 12.0\n",
      "2.0 * 8.0 = 16.0\n",
      "结果矩阵[2,2]:28.0\n",
      "\n",
      "手动推演结果:\n",
      "[[12. 14.]\n",
      " [24. 28.]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "# 定义两个简单的矩阵\n",
    "m1 = np.array([[1, 1], [2, 2]], dtype=np.float32)\n",
    "m2 = np.array([[5, 6], [7, 8]], dtype=np.float32)\n",
    "# m1 = m1.reshape(2,2)  # 矩阵1最后1个维度 == 矩阵2倒数第2个维度\n",
    "# m2 = m2.reshape(2,2)\n",
    "\n",
    "# 使用 np.dot 进行矩阵乘法\n",
    "result_dot = np.dot(m1, m2)\n",
    "\n",
    "# 使用 @ 运算符进行矩阵乘法\n",
    "result_at = m1 @ m2\n",
    "\n",
    "print(\"矩阵 1:\")\n",
    "print(m1)\n",
    "print(\"矩阵 2:\")\n",
    "print(m2)\n",
    "print(\"使用 np.dot 得到的矩阵乘法结果:\")\n",
    "print(result_dot)\n",
    "print(\"使用 @ 运算符得到的矩阵乘法结果:\")\n",
    "print(result_at)\n",
    "\n",
    "# 创建一个全零矩阵，用于存储手动推演的结果\n",
    "# 结果矩阵的行数等于 matrix1 的行数，列数等于 matrix2 的列数\n",
    "manual_result = np.zeros((m1.shape[0], m2.shape[1]), dtype=np.float32)\n",
    "\n",
    "# 外层循环：遍历 matrix1 的每一行\n",
    "# i 表示结果矩阵的行索引\n",
    "for i in range(m1.shape[0]):\n",
    "    # 中层循环：遍历 matrix2 的每一列\n",
    "    # j 表示结果矩阵的列索引\n",
    "    for j in range(m2.shape[1]):\n",
    "        # 初始化当前位置的结果为 0\n",
    "        manual_result[i, j] = 0\n",
    "        # 内层循环：计算 matrix1 的第 i 行与 matrix2 的第 j 列对应元素的乘积之和\n",
    "        # k 表示参与乘法运算的元素索引\n",
    "        for k in range(m1.shape[1]):\n",
    "            # 打印当前正在计算的元素\n",
    "            print(f\"{m1[i, k]} * {m2[k, j]} = {m1[i, k] * m2[k, j]}\")\n",
    "            # 将 matrix1 的第 i 行第 k 列元素与 matrix2 的第 k 行第 j 列元素相乘，并累加到结果矩阵的相应位置\n",
    "            manual_result[i, j] += m1[i, k] * m2[k, j]\n",
    "        # 打印当前位置计算完成后的结果\n",
    "        print(f\"结果矩阵[{i+1},{j+1}]:{manual_result[i, j]}\\n\")\n",
    "\n",
    "print(\"手动推演结果:\")\n",
    "print(manual_result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.save('result.npy',manual_result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[12., 14.],\n",
       "       [24., 28.]], dtype=float32)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_np = np.load('result.npy')\n",
    "result_np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 7, 9])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3])\n",
    "b = np.array([4,5,6])\n",
    "a + b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 3],\n",
       "       [1, 3],\n",
       "       [2, 4],\n",
       "       [3, 5]])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([(1,2), (2,2), (3,3), (4,4)])  # shape(4,2)\n",
    "b = np.array([-1,1])  # shape(2)-> shape(1,2) -> shape(4,2)  [[-1,1],[-1,1],[-1,1],[-1,1]]\n",
    "a + b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
