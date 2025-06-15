一、深度学习基础
1. 什么是深度学习
一种机器学习方法，基于多层神经网络结构来进行数据表示与学习。

核心特点：自动特征提取、端到端训练、强泛化能力。

2. 人工神经网络（ANN）基础
感知机（Perceptron）：最基础的神经元模型，输入加权求和 + 激活函数。

激活函数：

Sigmoid：输出范围(0,1)，容易饱和，梯度消失；

Tanh：输出(-1,1)，中心化；

ReLU：非线性强，收敛快，最常用；

Leaky ReLU / GELU：解决ReLU死神经元问题。

3. 反向传播与梯度下降
前向传播：计算输出；

损失函数（Loss Function）：衡量预测与真实值差距；

反向传播（Backpropagation）：利用链式法则求导，更新权重；

优化算法：

SGD（随机梯度下降）；

Adam（常用，效果好）；

RMSprop, Adagrad等。

二、卷积神经网络（CNN）基础
1. 为什么用CNN？
避免全连接参数过多；

能有效捕捉局部特征；

参数共享、稀疏连接，提高效率与泛化能力。

2. 卷积层（Convolutional Layer）
卷积核（Filter）：在输入上滑动，提取局部特征。

参数：

核大小（如3×3）；

步长（Stride）；

填充（Padding）：Same / Valid；

输出尺寸计算：

𝑂=⌊(𝐼+2𝑃−𝐾)/𝑆⌋+1

其中，
I=输入尺寸，
K=核尺寸，
P=填充，
S=步长。

3. 池化层（Pooling Layer）
用于降低特征图尺寸、减少参数、防止过拟合。

常见方法：

最大池化（Max Pooling）；

平均池化（Average Pooling）。

4. 批归一化（Batch Normalization）
加速收敛，防止梯度消失；

对每一层的输出进行归一化处理。

5. 全连接层（Fully Connected Layer）
将提取的局部特征整合为最终分类/回归结果；

通常在CNN尾部接几层全连接 + softmax。

6. Dropout
防止过拟合；

随机“丢弃”部分神经元，提高模型鲁棒性。

三、CNN 常见结构
网络结构	特点
LeNet-5	最早用于手写数字识别
AlexNet	ImageNet竞赛冠军，使用ReLU与Dropout
VGGNet	统一使用3×3卷积核，结构简洁
GoogLeNet（Inception）	引入Inception模块，多尺度提取特征
ResNet	引入残差连接（skip connection），解决深层网络退化问题

四、训练与调参要点
1. 数据预处理
归一化；

数据增强（翻转、旋转、裁剪等）；

2. 损失函数选择
分类问题：交叉熵（Cross Entropy）；

回归问题：均方误差（MSE）。

3. 训练技巧
学习率衰减；

早停（Early Stopping）；

交叉验证；

使用预训练模型 + 微调（Fine-tuning）。

五、实践框架与工具
主流框架：TensorFlow / PyTorch / Keras

GPU训练：推荐使用CUDA + cuDNN

可视化工具：TensorBoard, matplotlib, seaborn