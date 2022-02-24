
# EM算法

* EM算法是一种迭代算法，用于含有隐变量的概率模型参数的极大似然估计，或极大后验概率估计。EM算法的每步迭代由两步组成：E步，求期望，M步，求极大，所以称为期望极大算法。

* EM算法与初值的选择有关，选择不同的初值可能得到不同的参数估计值。

#### EM算法

* 输入：观测变量数据Y，隐变量数据Z，联合分布$P(Y<Z|\theta)$，条件分布$P(Z|Y<\theta)$
* 输出：模型参数$\theta$

（1）选择参数的初值$\theta^(0)$，开始迭代；

（2）E步：记$\theta^(i)$为第i次迭代参数$\theta$的估计值，在第i+1次迭代的E步，计算

$$
Q(\theta,\theta^{(i)})=E_z[logP(Y,Z|\theta)|Y,\theta^{(i)}]
=\sum_{z}logP(Y,Z|\theta)P(Z|Y,\theta^{(i)})
$$ **（Q函数）**

（3）M步：求使$Q(\theta, \theta^{(i)})$极大化的$\theta$，确定第i+1次迭代的参数的估计值$\theta^{(i+1)}$

$$
\theta^{(i+1)}=arg\max_{\theta} Q(\theta,\theta^{(i)})
$$

（4）重复第（2）步和第（3）步，直到收敛。

* 1.参数的初值可以任意选择，但EM算法对初值是敏感的。
* 2.$Q(\theta,\theta^{(i)})$的第1个变元表示要极大化的参数，第2个变院表示参数的当前估计值，每次迭代实际在求Q函数及其极大。
* 3.M步求$Q(\theta,\theta^{(i)})$的极大化，得到$\theta^{(i+1)}$，完成一次迭代，后面将证明每次迭代使似然函数增大或达到局部极值。
* 4.给出停止迭代的条件，一般是对较小的整数$\xi_1$，若满足

$$
||\theta^{(i+1)}-\theta^{(i)}||<\xi_1
$$

则停止迭代。

### EM算法的收敛性

* 定理1：设$P(Y|\theta)$为观测数据的似然函数，$\theta^{(i)})$为EM算法得到的参数估计序列，$P(Y|\theta^{(i)})$为对应的似然函数序列，则$P(Y|\theta^{(i)})$是单调递增的，即

$$
P(Y|\theta^{(i+1)})\ge P(Y|\theta^{(i)})
$$

* 定理2：设$L(\theta)=logP(Y|\theta)$为观测数据的对数似然函数，$\theta^{(i)}$为EM算法得到的参数估计序列，$L(\theta^{(i)})$为对应的对数似然函数序列。

（1）如果$P(Y|\theta)$有上界，则$L(\theta^{(i)})=logP(Y|\theta^{(i)})$收敛到某一值$L^*$；

（2）在函数$Q(\theta,\theta^{'})$与$L(\theta)$满足一定条件下，由EM算法得到的参数估计序列$\theta^{(i)}$的收敛值$\theta_*$是$L(\theta)$的稳定点。

### EM算法在高斯混合模型学习中的应用

#### 高斯混合模型

* 定义：高斯混合模型是指具有如下形式的概率分布模型：

$$
P(y|\theta)=\sum_{k=1}^{K}\alpha_k \phi(y|\theta_k)
$$

$\alpha_k$是系数，$\phi(y|\theta_k)$是高斯分布密度。

#### 高斯混合模型参数估计的EM算法

* 输入：观测数据$y_1,y_2,...,y_N$，高斯混合模型；
* 输出：高斯混合模型参数

（1）取参数的初始值开始迭代；

（2）E步：依据当前模型参数，计算分模型k对观测数据$y_i$的响应度

$$
\hat{\gamma_{jk}}=\frac{\alpha_k \phi (y_j|\theta_k)}{\sum_{k=1}^{K} \alpha_k \phi(y_j|\theta_k)}
$$

（3）M步：计算新一轮迭代的模型参数

$$
\hat{\mu_k}=\frac{\sum_{j=1}^{N}\hat{\gamma_{jk}}y_j}{\sum_{j=1}^{N}\hat{\gamma_{jk}}}
$$

$$
\hat{\sigma_k^2}=\frac{\sum_{j=1}^{N}\hat{\gamma_{jk}(y_j-\mu_k)^2}y_j}{\sum_{j=1}^{N}\hat{\gamma_{jk}}}
$$

$$
\hat{\alpha_k}=\frac{\sum_{j=1}^{N}\hat{\gamma_{jk}}}{N}
$$

（4）重复第（2）步和第（3）步，直到收敛。

### EM算法的推广

#### 1.F函数的极大-极大算法

#### 2.GEM算法

* 每次迭代增加F函数值，从而增加似然函数值。

### 代码实现

#### E step：


```python
import numpy as no
import math
```


```python
pro_A, pro_B, pro_C = 0.5, 0.5, 0.5

def pmf(i, pro_A, pro_B, pro_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow((1 - pro_B), 1 - data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow((1 - pro_C), 1 - data[i])
    return pro_1 / ( pro_1 + pro_2)
```

#### M step:


```python
class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow(
            (1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow(
            (1 - self.pro_C), 1 - data[i])
        return pro_1 / (pro_1 + pro_2)
    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1 / count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum(
                [_pmf[k] for k in range(count)])
            pro_C = sum([(1 - _pmf[k]) * data[k]
                        for k in range(count)]) / sum([(1- _pmf[k])
                                                       for k in range(count)])
            print('{}/{} pro_a:{:.3f},pro_b:{:.3f}, pro_c:{:.3f}'.format(
                d + 1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
```


```python
data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
```


```python
em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)
```

    init prob:0.5, 0.5, 0.5
    


```python
# 第一次迭代
f.send(1)
```

    1/10 pro_a:0.500,pro_b:0.600, pro_c:0.600
    


```python
# 第二次迭代
f.send(2)
```

    2/10 pro_a:0.500,pro_b:0.600, pro_c:0.600
    


```python
em = EM(prob=[0.46, 0.55, 0.67])
f2 = em.fit(data)
next(f2)
```

    init prob:0.46, 0.55, 0.67
    


```python
f2.send(1)
```

    1/10 pro_a:0.462,pro_b:0.535, pro_c:0.656
    


```python
f2.send(2)
```

    2/10 pro_a:0.462,pro_b:0.535, pro_c:0.656
    
