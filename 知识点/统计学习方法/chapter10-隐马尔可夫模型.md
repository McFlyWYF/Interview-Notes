
# 隐马尔可夫模型

* 隐马尔可夫模型是可用于标注问题的统计学习模型，描述由隐藏的马尔可夫链随机生成不可观测的状态的序列，再由各个状态随机生成一个观测而产生观测序列的过程，属于生成模型。

* 隐马尔可夫模型由初始状态概率向量$\pi$、状态转移概率矩阵A和观测概率矩阵B决定。因此，隐马尔可夫模型可以写成$\lambda=(A,B,\pi)$。

* 隐马尔可夫模型是一个生成模型，表示状态序列和观测序列的联合分布，但是状态序列是隐藏的，不可观测的。

* 隐马尔可夫模型可以用于标注，这时状态对应着标记。标注问题是给定观测序列预测其对应的标记序列。

* 隐马尔可夫模型是关于时序的概率模型，描述由一个隐藏的马尔可夫链随机生成不可观测的状态随机序列，再由各个状态生成一个观测而产生观测随机序列的过程。隐藏的马尔可夫链随机生成的状态的序列称为状态序列；每个状态生成一个观测，由此产生的观测的随机序列称为观测序列。

* 隐马尔可夫模型由初始概率分布、状态转移概率分布以及观测概率分布确定。

### 隐马尔可夫模型的3个基本问题：
* （1）概率计算问题。给定模型和观测序列，计算在模型下观测序列出现的概率。
    * 给定模型$\lambda=(A,B,\pi)$和观测序列$O=(o_1,o_2,...,o_T)$，计算在模型$\lambda$下观测序列O出现的概率$P(O|\lambda)$。前向-后向算法是通过递推地计算前向-后向概率可以高效地进行隐马尔可夫模型的概率计算。
* （2）学习问题。已知观测序列，估计模型参数，使得在该模型下观测序列概率最大。即用极大似然估计的方法估计参数。
    * 已知观测序列$O=(o_1,o_2,...,o_T)$，估计模型$\lambda=(A,B,\pi)$参数，使得在该模型下观测序列概率$P(O|\lambda)$最大。即用极大似然估计的方法估计参数。B-W算法，也就是EM算法可以高效地对隐马尔可夫模型进行训练。它是一种非监督学习算法。
* （3）预测问题。已知模型和观测序列，求对给定观测序列条件概率最大的状态序列，即给定观测序列，求最有可能的对应的状态序列。
    * 已知模型$\lambda=(A,B,\pi)$和观测序列$O=(o_1,o_2,...,o_T)$，求对给定观测序列条件概率$P(I|O)$最大的状态序列$I=(i_1,i_2,...,i_T$。维特比算法应用动态规划高效地求解最优路径，即概率最大的状态序列。
