# HPXNN
HPX, short for High Performance ParalleX, is a runtime system for high performance computing. 
The project aims at implementing HPX in the C++ backends of various machine learning libraries with the help of a converter tool that has been built to achieve considerable performance improvements. The machine learning libraries used in the project to test HPX are Pytorch and OpenNN. OpenNN is a software library written in C++, that is used for the implementation of neural networks in the area of deep learning research. The project utilises the HPX runtime system to run individual tasks on the system cores to achieve better execution speeds.

The Topics of focus of this directory is HPX, Machine Learning, C++, OpenNN, PyTorch.

The contirbutors to this project are 
<ul>
    <li>Amritesh [PES1UG19CS056]</li>
    <li>Ashwath Krishnan [PES1UG19CS199]</li>
    <li>Srikar S [PES1UG19CS505]</li>
</ul>
This project was done under the mentorship of <b>Navraj Singh</b> and was guided by <b>Dr. Rahul Nagpal</b>. <br>
This project was a done in association with "Center of Advanced Parallel Systems - Parallel System Research Lab" (CAPS-PSRL), PES University, Banashankari, Bengaluru, Karnataka 560085.

## Contents of the Directory

This directory has major sub-folders on which HPX was tested. 
<ol>
    <li>Tensor Functions</li>
    <li>Optimization</li>
    <li>Autograd</li>
    <li>MNIST</li>
    <li>CNN Program -resnet 50</li>
    <li>Linear Regression</li>
    <li>Logistics Regression</li>
    <li>2-Layer Neural Network</li>
    <li>Feed Forward Neural Network</li>
    <li>hpx_pytorch_convertor.cpp</li>
    <li>OpenNN Library</li>
</ol>

## Description
<ol>
    <li><b>Tensor Functions</b>: This includes simple tensor functions and their implementation using HPX.
    <li><b>Optimization</b>: The program mimics the working of an optimizer(Adam Optimizer) in Neural Networks.
    <li><b>Autograd</b>: This program runs basic autograd operations with some higher order gradient examples and some of the custom autograd functions.
    <li><b>MNIST</b>: The Program works on the dataset of hand-written digits and displays the loss and accuracy at the end of each epoch.
    <li><b>CNN Program -resnet 50</b>: This program implements ResNET50 (CNN) classifier, which  identifies objects in images.
    <li><b>Linear Regression</b>: A simple Linear Regression program.
    <li><b>Logistics Regression</b>: A simple Logistic Regression program.
    <li><b>2-Layer Neural Network</b>: A fully-connected ReLU network with one hidden layer, trained to predict y from x by minimizing squared Euclidean distance.
    <li><b>Feed Forward Neural Network</b>: It is a biologically inspired classification algorithm. It takes the input, feeds it through several layers one after the other, and then finally gives the output.
    <li><b>hpx_pytorch_convertor.cpp</b>: This is the converter tool responsible for converting C++ PyTorch programs to HPX C++ PyTorch Programs.  
    <li><b>OpenNN Library</b>: OpenNN stands for Open Neural Networks. It is a software library written in the C++ programming language. It implements a major area of machine learning, neural networks. Written in C++ for advanced analytics, OpenNN’s main advantage is its high performance. 
</ol>
This library implements the HPX backend on the above listed files. Files like Tensor Functions, Optimization, Autograd, MNIST, CNN-Resnet 50, Linear Regression, Logistics Regression, 2-Layer Neural Network, Feed Forward Neural Network are common machine learning algorithms which were used for testing HPX's performance on PyTorch Programs. Changes like asynchronous function calls using async() and futures were used and standard for-loops were converted to HPX for-loops.<br>
HPX_CONVERTER.cpp is a converter tool responsible for converting C++ PyTorch programs into HPX implemented C++ PyTorch programs. <br>
The OpenNN Library contains the OpenNN repository with HPX backend implemented. The testing has been done on example files like breast_cancer, airfoil_self_noise, airline_passengers, iris_plant, logical_operations, simple_function_regression. Moreover, changes have been made to opennn module files like adaptive_moment_estimation.cpp, data_set.cpp and neural_network.cpp,etc. 

> Each of these files contain the absolute path to either the HPX library or the PyTorch Library. These paths must be changed accordingingly on each system.<br>
> These paths have mostly been mentioned in the CMakeLists.txt file with each folder. Some files also contain the absolute path to their datasets.<br> <i><b>Please make the necessary changes.</b></i>

## Execution of the Files

PyTorch Examples files can be individually executed by the running the following codes:
```
$ mkdir build && cd build
$ cmake -DHPX_DIR=[PATH TO FOLDER CONTAINING HPXConfig.cmake] .. 
$ make
```
The make command speed can be increased by using the -jN flag where N is the Number of threads. 
<br>
Each file has a different executable name such as 
autograd is ./autograd
<br>
These names can be checked in their respective CMakeLists.txt
To run the executable, enter the build directory.
```
$ ./[EXECUTABLE_NAME]
```
<br> To run the CNN Program -resnet 50 program after the execution of the cmake and make command, follow the following steps:
```
$ ./predict_demo ../resnet50.pt ../label.txt          
```

### Running of OpenNN HPX Backend
Inside the opennn directory follow these commands listed below
```
$ mkdir build && cd build
$ cmake -DHPX_DIR=/PATH/TO/FOLDER/CONTAINING/HPXConfig.cmake/ .. 
$ make
```
To Run the examples in the OpenNN Directory, follow the commands listed below (inside the build directory):
```
$ cd examples/[EXAMPLE_NAME]
$ ./[EXECUTABLE_NAME]
```
