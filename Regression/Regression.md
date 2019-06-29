# Regression Analysis

Regression analysis is a predictive modeling technique which investigates the relationship among variables. It aims to find the **_causal effect relationship_** between a dependent variable ('target' or 'label') and one or more indenpendent variables ('predictors').

Regression analysis is an important tool for analyzing data. Here, we fit a curve to the data points, while the difference between the distaces of data points from the curve is mimimized (try to make the curve as close as possible to the original data points).

Below is an example of how regression used among stock analysis.

![Linear Regression for Stock](https://www.analyticsvidhya.com/wp-content/uploads/2015/08/Regression_Line.png)

By using the regression analysis, two main insights can be observed:

* The relationships between dependent variable and independent variable.
* The strength of impact of multiple independent variables on the dependent variable.

Regression analysis also allows us to compare the effects of variables measured on different scales, such as the BMI and body weights. In that case, one can also use regression analysis to select the best set of variables used for building predictive models.

## Regression Models

* [Linear Regression](#Linear-Regression)
* [Logistic Regression](#Logistic-Regression)
* [Ridge Regression](#Ridge-Regression)
* [LASSO Regression](#LASSO-Regression)

----
### Linear Regression
Below is a brief summary of Linear Regression. For more details, please check:
* [Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
* [Linear Regression Implementation - GeeksforGeeks](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

Linear Regression might be the most popular statistical models in regression analysis. It allows us to learn a function which can represent the relationship between some data points _x_ and corresponding _y_. Such function/relationship is also called hypothesis.

_h(x) = W*x + b_

where _W_ is the parameter of weights (vector) and _b_ is the parameter of bias (scalar). All we need to do next is to estimate the value of _W_ and _b_ from the give data such that the hypothesis _h(x)_ is as close as possible to the original data point _y_. Therefore, the loss/cost function is introduced.

![Loss Function](http://www.sciweavers.org/tex2img.php?eq=%20J%28W%2C%20b%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%28y_i%20-%20h%28x_i%29%29%20%5E%202%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
 
where _n_ is the number of data points in the given dataset. This cost function is also called **Mean Squared Error**.

For finding the optimized value of the parameters for which J is minimum, a commonly used optimizer algorithm, **Gradient Descent** can be used here.

### Logistic Regression

### Ridge Regression

### LASSO Regression

## Summary
