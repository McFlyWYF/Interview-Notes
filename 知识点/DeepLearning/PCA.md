
# PCA（主成分分析）降维

* PAC是一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量。

* 如果基的数量少于向量本身的维数，则可以达到降维的效果。

* 问题转化为：寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，方差值最大。

* 对于高维数据，我们用协方差进行约束，协方差表示两个变量的相关性。为了让两个变量尽可能表示更多的原始信息，我们希望他们之间不存在线性相关性，因为相关性意味着两个变量不是完全独立，必然存在重复表示的信息。

* 将一组N维向量降为K维，其目标是选择K个单位正交基，使得原始数据变换到这组基上后，各变量两两间协方差为0，而变量方差则尽可能大（在正交的约束下，取最大的K个方差）。

* 寻找一个矩阵P，满足$PCP^T$是一个对角矩阵，并且对角元素按从大到小依次排列，那么P的前K行就是要寻找的基，用P的前K行组成的矩阵乘以X就使得X从N降维到了K维并满足上述优化条件。

### 求解步骤

设有m条n维数据。

* 将原始数据按列组成n行m列矩阵X；
* 将X的每一行进行零均值化，即减去这一行的均值；
* 求出协方差矩阵$C=\frac{1}{m}XX^T$；
* 求出协方差矩阵的特征值及对应的特征向量；
* 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P；
* $Y=PX$即为降维到k维后的数据。

### 性质

* **缓解维度灾难**：PCA算法通过舍去一部分信息之后能使得样本的采样密度增大。
* **降噪**：当数据受到噪声影响时，最小特征值对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到降噪的效果；
* **过拟合**：PCA保留了主要信息，但这个主要信息只是针对训练集的，而且这个主要信息未必是重要信息。有可能舍弃了一些看似无用的信息，但是这些看似无用的信息恰好是重要信息，只是在训练集上没有很大的表现，所以PCA可能加剧了过拟合；
* **特征独立**：PCA不仅将数据压缩到低维，它也使得降维之后的数据各特征相互独立；

### 细节

#### 零均值化

* 当对训练集进行PCA降维时，也需要对验证集、测试集执行同样的降维。而对验证集、测试集执行零均值化操作时，均值必须从训练集计算而来，不能使用验证集或测试集的中心向量。
