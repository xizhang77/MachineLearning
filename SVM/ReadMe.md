# Support Vector Machine

In Machine Learning area, Support Vector Machine (SVM) is a supervised learning method that analyze data for classification. It aims to find the **_hyperplane_** which can seperate data into different regions. 

When data is linear sperateable, SVM can be reduced to **Linear Classification**. When data is not linear separable, **kernel trick** is used to project data points to a higher dimensional space until the data is "linear separable". Below is a simple explanation of how kernel works.

![Kernel Trick](https://miro.medium.com/max/1400/1*3t_Gn5yuirT6fSC-sbxKAA.png)

Here is a brief review of different SVM methods. For more details, please check:
* [Support Vector Machine - Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine#Linear_SVM)
* [Support Vector Machines Explained](https://medium.com/@zachary.bedell/support-vector-machines-explained-73f4ec363f13)
* [TensorFlow Cookbook - 04_SVM](https://github.com/nfmcclure/tensorflow_cookbook/tree/master/04_Support_Vector_Machines)


## SVM Models

* [Linear SVM](#Linear-SVM)
* [Non Linear SVM](#Non-Linear-SVM)

----
### Linear SVM
The code can be find in [Python Code](./code/LinearSVM.py) (implemented using TensorFlow).

### Non Linear SVM
