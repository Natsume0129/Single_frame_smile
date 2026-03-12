## 1
这周主要的内容是，在我们之前说的方法，也就是找一条真笑轴，看看其他笑容是如何沿着真笑轴走的，去计算他的projection和derivation的距离。

我今天会从假说开始，说明我的计算方法，结果，以及我的结论和想法

## 2

首先是我们的假设。

different types of smile have different temporal trajectory pattern in linear space.

Temporally，smiles are kinds of pattern changing from a non-smile region to a smile region in linear space. 

The trure-smile can provide a reference dynamic direction to measure how much other smiles advance along the main direction and deviate from the true smile path. 

## 3

基于这些假设，我们通过vgg-face fc7提取出原始的特征，然后经过预处理和重采样对齐之后，得到这个序列。

由20个4096维的向量组成

然后考虑到，Comparing multiple data pairs are inappropriate, so it’s necessary to find a typical case (prototype trajectory) for each class

我主要用了两个方法，一个是用中位数计算出一条统计学上的典型曲线

另一个是采用计算序列之间的距离，选择总距离最小，也就是相对最中心位置的真实曲线的方法。

## 4
这里是方法A的计算方法，
对于normalized之后的每一个时间点，针对每一个维度，取所有序列中的中间数，然后把所有维度的中间数拼接成这个时间点的向量，逐时间点进行这个操作，最后得到prototype trajectory

## 5
然后方法B，我们上次说，我们使用的例子，最好能够有实际能够对照的例子，我已我选择了这个方法。
把序列的20个向量拼接成一个矩阵。
对于两条序列之间的距离，我们用Frobenius norm来计算。
从一条序列出发到所有其他序列距离之和最小的序列，就是定义为我们用来分析的prototype trajectory实例

## 6
这里是通过算法得到的三个class的prototype，他们看起来有不小的差异，不过因为序列内的光照和姿态大致是一致的， 所以在我们预处理的时候，这些共同语义会被削减，留下代表和第一帧的差异差分向量。

## 7
接下来是有关微笑主轴的定义，简单来说，向量g，是true smile 的prototype的首尾相连构成的向量，u则是代表方向的单位向量
如图所示，黑色曲线代表true smile的curve，我们连接他的头尾，构成main axis，

## 8
有了微笑主轴之后，我主要做3件事情。
一个是计算序列每一个时间点上在空间中的距离，对于两条序列，同一个时间点的向量的差分向量的模长意味着他们在线性空间中的大小，这里我用的Euclidean distance
图里，黑色曲线代表真笑的prototype，红色curve代表polite smile 的prototype，我计算了他们每一个时间点的绝对距离。

另外两个就是计算其他的类别的笑容，沿着true smile axis 走了多少，以及有多少偏移的计算

## 9
如图所示
图表的横轴代表是时间，因为重采样了20个点，所以最大值是20.
纵轴是代表距离，距离越大说明在空间中两点的表情的综合差异越大。
这里的anchor指的是被对比的序列。
比如左上角这张图，就是使用方法A得到的prototype，每一个时间点上，polite smile和 ambiguous smile的prototype距离true smile的prototype的距离

对于这个的结果，我觉得可以主要构成以下三点：
The initial distances are typically small, which is more likely to reflect similarities in a neutral state than differences in smiles themselves.
As time progresses, the distances between categories generally increase, indicating that dynamic differences become more pronounced after the smile unfolds.
Polite and ambiguous smiles are generally closer to each other, while both are clearly distinct from truesmiles.
## 10
接下来是，我想看看，不同的笑容是怎么沿着smile axis前进的？
对于一个时间节点，我需要先计算他相对于开始位置的差分向量。
然后计算他和笑容主轴的投影。
计算这个投影的模长占据笑容主轴向量g的比值，就能看到每个事件点沿着主轴走了多少。
## 11

## 12
方法A和方法B得到的结果如下图。
上面的图表是只包含prototype的
下面的图表，同时计算了所有参与计算的实例的结果。
虚线代表计算结果的均值。
浅色区域代表主要代表中间50%的样本区间
## 13
我的结论如下
When using the median calculation method in Method A, most true smile data does indeed progress along the main axis.
In Method B, although this value is much smaller, numerical changes still exist.

Regardless of Method A or Method B, the direction of progression of a "polite smile" over time differs significantly from that of a true smile.

Consistent with our definition, an "ambiguous smile" includes smiles that we cannot clearly categorize, so it should be a transitional state between a "polite smile" and a true smile.
Data-wise, an "ambiguous smile" does indeed lie between a true smile and a "polite smile," presenting an intermediate state.
## 14
最后，我计算了例子偏离真笑主轴多少，方法就是构造投影向量和原向量之间的差分向量，计算差分向量的模长。
这里的比例是计算了差分向量的模长和真笑主轴长度的比例
## 15
同样的，图表的横轴代表时间，纵轴代表比例，
可以看到，prototypes从0出发，比例都会逐渐变大，这意味着他的实际偏移量都是越来越大的。
除了真笑的最后一个时间点，因为这是从定义出发的
我觉得结论可以是以下的内容：
The truesmile still shows significant deviation from the true-smile axis in its middle stages.
This indicates that the truesmile's trajectory is not a straight line; the line connecting its beginning and end is only a rough reference direction.
The polite and ambiguous stages progress less along the along-axis, but their off-axis deviation is considerable, suggesting that they are not static, but rather changing primarily in other directions.
## 16
综上所述，我觉得现阶段的结论可以是下面这些：
Different smile trajectories do indeed differ in space.

Expressions in the neutral phase are more similar; as the smile intensifies, the similarity decreases.

Polite and ambiguous smiles move towards other regions in the feature space rather than going to true smile region.

Polite and ambiguous smiles are closer together.

The true smile shows a significant difference from the other two categories.