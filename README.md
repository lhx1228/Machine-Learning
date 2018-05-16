# 机器学习第一次编程练习

标签（空格分隔）： 机器学习

---

## 遇到的问题

### 梯度下降

在实现梯度下降时，最初实现的方式是类似于这样：

```
for iter = 1:num_iters
 
    theta(1) = theta(1) - alpha / m * sum(X * theta_s - y);      
    theta(2) = theta(2) - alpha / m * sum((X * theta_s - y) .* X(:,2));   
    theta_s=theta; 
```

但是总觉得麻烦，后来就想可不可以一行代码实现，后来经过摸索，上诉代码可以用下面的两行来替换：

```
for iter = 1:num_iters
    theta = theta - alpha * (X'*(X*theta - y)) / m;
```

### 特征缩放

其实最初一直不理解这个地方，觉得缩放之后得出的结果怎么可能与不缩放得出的结果相同，这次正好借这个机会实践证实了一下。

首先我们看不使用特征缩放的例子:

```
data = load('ex1data2.txt');
X = data(:,1:2);
y = data(:,3);
m = length(y);
X = [ones(m,1),X];
theta = [0;0;0];
alpha = 0.01;
for i = 1:10000,
    theta = theta - alpha * (X'*(X*theta-y)) / m;
end;
disp(theta);

disp(pinv(X'*X)*X'*y);
```

结果为:

```
   NaN
   NaN
   NaN
   
   89597.90954
     139.21067
   -8738.01911
```


很明显，这个结果不是我们想要的，因为其算出来的值与根据正规方程算出来的结果并不相同。

接下来我们看使用特征缩放之后的代码：

```
data = load('ex1data2.txt');
X = data(:,1:2);
X = (X - mean(X)) ./ std(X);
y = data(:,3);
m = length(y);
X = [ones(m,1),X];
theta = [0;0;0];
alpha = 0.01;
for i = 1:10000,
    theta = theta - alpha * (X'*(X*theta-y)) / m;
end;
disp(theta);

disp(pinv(X'*X)*X'*y);

```

结果如下：

```
   340412.65957
   110631.05028
    -6649.47427
    
   340412.65957
   110631.05028
    -6649.47427
```

结果与正规方程算出的相同。

### 多元线性回归预测结果不同

在多元线性回归问题中，我根据梯度下降和正规方程两种方法算出来theta值，可是最后两种方法预测出的结果不同，如下图：

![显示](https://raw.githubusercontent.com/lhx1228/Machine-Learning/master/the%20frist%20program%20pratice/Machine-Learning_1.png)

这是什么原因呢？

原来在**梯度下降方法中我使用了特征缩放，而在预测结果是，我所输入的数据并没有缩放，**所以导致预测出来的结果较大。

更改方法为将**需要进行预测的数据按照特征缩放的比例缩小。**例如需要测试的数据为`[1 1650 3]`，我们可以将它更改为`[1 (1650 - mu(1)) ./ sigma(1) (3 - mu(2)) ./ sigma(2)]`。 其中**mu为X的平均值，sigma为X的标准差。**


## 代码(octave/matlab实现)

想要了解代码的可以[点击这里](https://github.com/lhx1228/Machine-Learning)
