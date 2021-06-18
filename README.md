# CS-439-Project

Mini-Project for the course [Optmization for Machine Learning](https://github.com/epfml/OptML_course) (description [here](https://github.com/epfml/OptML_course/blob/master/labs/mini-project/miniproject_description.pdf)).

Topic:
 * Meta-Learning/AutoML: Learning the learning rate. 
---

In this project we have implemented the algorithm *WNGrad* presented by [Léon Bottou et. al.](https://arxiv.org/pdf/1803.02865.pdf) and compared its performance against other standard, gradient-based optimization algorithms. The main advantage of *WNGrad* over other algorithms like *SGD* or *Adagrad* is that, if there is a correct initialization, there is no hyperparameter to fine-tune -a process which can be very resource consuming-. We tried different problems, including convex problems (least squares, support vector machines) and non-convex problems (dense neural networks, convolutional neural networks). 

Furthermore, after realizing that the parameter update proposed by [Léon Bottou et. al.](https://arxiv.org/pdf/1803.02865.pdf) was too big and could truncate learning, we proposed a modification to the update rule, that proved to outperform *WNGrad*, and in some scenarios, a properly tuned *SGD*. 

![image](https://user-images.githubusercontent.com/65513243/122571189-c3713400-d04c-11eb-9e57-93f20e793792.png)


The convex and non-convex settings were performed in `Python 3.9`, using `PyTorch 1.8.0` as core library. Experiments were performed on `iPython` notebooks.The Riemannian setting was performed in MatLab 2020b using, the manopt toolbox created and maintained by Boumal et al. 

To reproduce the experiments performed in Python (Least Squares, SVM, and a CNN), run [`code/Main.ipynb`](https://github.com/Alejandro-1996/CS-439-Project/code/Main.ipynb). To reproduce the experiments in the Riemannian setting, run [`code/rayleighquotient/main.m`](https://github.com/Alejandro-1996/CS-439-Project/code/rayleighquotient/main.m) and [`code/robustsubspaceanalysis/main.m`](https://github.com/Alejandro-1996/CS-439-Project/code/robustsubspaceanalysis/main.m).
