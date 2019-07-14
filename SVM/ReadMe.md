# Support Vector Machine

In Machine Learning area, Support Vector Machine (SVM) is one of the most important supervised learning methods that analyze data for classification. It aims to find a **_hyperplane_** which can seperate data into different regions. 

When data is linearly separable, SVM is simplified to **Linear Classification**. When data is not linearly separable, **kernel trick** is used to project data points to a higher dimensional space such that the data is "linear separable". Below is a simple explanation of how kernel works.

<img src=https://miro.medium.com/max/1400/1*3t_Gn5yuirT6fSC-sbxKAA.png width="600">


Below are some good references for SVM with detailed explanation and derivation.
* [Support Vector Machine - Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Linear_SVM)
* [Support Vector Machines Explained](https://medium.com/@zachary.bedell/support-vector-machines-explained-73f4ec363f13)
* [TensorFlow Cookbook - 04_SVM](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines)


## SVM Models

* [Linear SVM](#Linear-SVM)
* [Non Linear SVM](#Non-Linear-SVM)
* [Muti-class SVM](#Multi-Class-SVM)
----
### Linear SVM
The code can be find in [Python Code](./code/LinearSVM.py) (implemented using TensorFlow). In linear SVM, one is trying to find a hyperplane
```
wx - b = 0
```
such that, all the data points that belong to positive class lie on the positive side (same direction as normal vector _w_), and all the data points that belong to negative class lie on the negative side (opposite direction to normal vector _w_).

<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png width="300">

When the data is linearly separable, one can also select two hyperplanes such that two classes of data are seperated as much as possible. This methodology is so called **Hard Margin** (see yellow zone in the above figure). Therefore, the objective function is re-written as

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/94c99827acb10edd809df63bb86ca1366f01a8ac)

To extend the linear SVM to non-linear cases, **_Hinge Loss_** is introduced later. 

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/f5f7d77f3d46cac51fbac58545aa1a1a183fdf7f)

When data lies on the correct side of the margin, _1 - y(wx-b) < 0_ and the hinge loss is equal to 0. When data on the wrong side of the margin, the function's value is proportional to the distance from the margin.

Therefore, the objective function is written as

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/53b729df53f32c7fbf933b1b034a8e368037d9b5)

and can be simplified by solving for the Lagrangian **dual**

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/6521b9d3e009bca40552bb94d204a4da1f2af4fe)

Below is an example of how linear SVM used among classification problem (data used: Iris.)

<img src="./figure/line_svm.png" width="500">

### Non Linear SVM
The code can be find in [Python Code](./code/KernelSVM.py) (implemented using TensorFlow).

<img src="./figure/nline_svm.png" width="500">


### Multi Class SVM
