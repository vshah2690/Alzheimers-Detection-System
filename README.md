# Alzheimers-Detection-System

Project Name : Acquiring Alzheimer’s Syndrome Detection through the Application of Machine Learning.
Organization : University of East London
Purpose : Dissertation
Course : MSc in Computer Science
Duration : May,2023 - September,2023


**ABSTRACT**

A degenerative brain ailment that has a severe impact on both individuals and their families, Alzheimer's disease is a major public health concern. For early interventions and customised healthcare, Alzheimer's must be accurately and promptly diagnosed. Utilising neuroimaging and clinical data, machine learning techniques have recently produced intriguing findings in helping with the recognition and forecasting of Alzheimer's disease. The most recent and cutting-edge machine learning techniques used in Alzheimer's disease detection are highlighted in this in-depth assessment. The research findings deliver particulars on how a variety neuroimaging techniques, including magnetic resonance imaging (MRI), positron emission tomography (PET), and functional MRI (fMRI), help in acquiring the course of the disease and figuring guaranteed biomarkers. A comprehensive review of the machine learning techniques and algorithms used to identify Alzheimer's disease is delivered. It explores cutting-edge deep learning models like convolutional neural networks (CNNs) and recurrent neural networks (RNNs), in addition to well-known supervised learning techniques like support vector machines (SVM), logistic regression, and decision trees. The analysis not only evaluates the strengths, flaws, and comprehending of each model individually but also looks at how ensemble approaches and transfer learning tactics might help the models perform better as a whole. The greatest method for diagnosing Alzheimer's disease is magnetic resonance imaging (MRI), which is one of the technologies now in use. The early stages of AD make it difficult to notice even the smallest changes in the brain. In this paper, a number of MRI image dataset is utilized to create models powered by deep learning for Alzheimer's identifying purposes. Images of brains in a variety of phases of deceit and absurdity make up the dataset. Convolutional Neural Network (CNN), ResNet50V2, and Visual Geometry Group16 (VGG16) and so on are the deep learning models that can be employed in the study. The customized CNN in which the dataset is iterated gives 98.85% and loss of 0.036. While the accuracy obtained from other pre-defined architectures like ResNet50V2 and VGG16 are 64.17% and 88.25% respectively. 

-> Keywords: Alzheimer’s Disease; Neurodegenerative Disorder; Convolutional Neural Network (CNN); Machine Learning Algorithms; MRI – magnetic resonance imaging; Data Augmentation and Data Preprocessing.


**Dataset Description**

The dataset consists of four picture classes in both the Training and Testing sets, which contains MRI scans of brain namely: 

|Class Name         | Total Number of Images
|1 NonDemented      |    3200
|2 MildDemented     |    896
|3 VeryMildDemented |    2240
|4 ModerateDemented |    64

1. Non Demented Brain Images:
 Magnetic resonance imaging (MRI) images of non-demented brain pictures 
frequently indicate no notable irregularities.

2. Mild Demented:
 Brain scans of people with mild dementia often reveal more obvious brain 
shrinkage or morphological alterations.

3. Very Mild Demented:
 Very mild dementia brain imaging may reveal minor alterations, such as minor 
atrophy or smaller volumes within particular brain areas.

4. Moderate Demented:
 A large amount of brain shrinkage and harm to the structure can be detected in 
moderately demented brain visuals.

**Requirements**

Tools that has been used to achieve the goal of this project: 

 Hardware Resources:
- A proper computing infrastructure
- Sufficient CPUs/GPUs
- Sufficient memory (RAM)

 Software Resources:
- Programming Language: Python
- Machine Learning Libraries: TensorFlow or PyTorch
- Image Processing Libraries: OpenCV and PIL
- Development Environment: Jupyter Notebook or PyCharm
- Data Visualization Library: Matplotlib or Seaborn
or
- Visualization Tool: Microsoft Power BI

The images from the dataset are processed using the image processing libraries as it is very crucial to prepare the data for machine learning model. OpenCV and PIL (Python Imaging Library) can be utilized to handle the image data and perform tasks like resizing, cropping and normalization. The development environments like Jupyter Notebook or PyCharm can be used to run and debugging of the code. Then the Data Visualization will be achieved with the use of libraries; Matplotlib or Seaborn or a visualization tool; Power BI to determine the prediction results.

**Prerequisite Knowledge/skills Required**

-> Fundamentals of Machine Learning
-> Basic Knowledge of Deep Learning
-> Python Programming 
-> Data & Image Processing
-> Evaluation Metrics
-> Data Visualization Techniques
-> Some Domain Knowledge

**Directories and Files with their Brief**

1. Code Files :- 
    - Contains the code files of the project.

        1. CodeUsingRMSpropOptimizer.py 
            - Contains code using customised CNN Architecture and RMSprop optimizer.
        2. CodeUsingSGDOptimizer.py
            - Contains code using customised CNN Architecture and SGD optimizer.
        3. Customised_Code_Final.py
            - Contains code using customised CNN Architecture and Adam optimizer.
        4. ResNet50V2.py
            - Contains code using ResNet50V2 Architecture and Adam optimizer.
        5. VGG16.py
            - Contains code using VGG16 Architecture and Adam optimizer.
2. Dataset :-
    - Contains the Dataset link of kaggle.
  
3. Documents :-
    - Contains the documents related to the project.

        - Initial Project Proposal
        - Powerpoint Presentation
        - Report


**Code Information and Result Overview**

The dataset is gone under training of different architecture of the neural network. The architecture utilized are as under:
[1] CNN (Custom)
[2] ResNet50V2
[3] VGG16

The different types of optimizer are also being used in the model getting the highest accuracy to achieve more and more accuracy.
[1] ADAM
[2] SGD
[3] RMSprop

The below table represents different outcomes when the dataset is trained using different combination of architecture and optimizers.


    |Architecture | Optimizer | Accuracy | Loss
    ---------------------------------------------
1.  |Custom       | Adam      |  98.85%  | 0.036
2.  |Custom       | RMSprop   |  97.36%  | 0.077
3.  |Custom       | SGD       |  64.57%  | 0.798
4.  |ResNet50V2   | Adam      |  64.17%  | 0.824
5.  |VGG16        | Adam      |  88.25%  | 0.345

- By doing a comparative analysis of the five distinct architectures utilizing different optimizers, we have evaluated their performance in terms of accuracy and loss metrics. Now, let us proceed with the analysis of the obtained results. 
  
- The architecture that exhibits the highest level of accuracy is the Custom Architecture, which utilizes the Adam Optimizer. This particular architecture achieves a notable accuracy rate of 98.85%. This observation implies that the utilization of the customized architecture in conjunction with 
the Adam optimizer yielded favourable outcomes for the given dataset. The observed low loss value of 0.036 suggests that the training process has been efficient and that the model has converged effectively.

- In addition, the architecture, Custom using the RMSprop optimizer, attained a classification accuracy of 97.36% and a corresponding loss value of 0.077. The performance of the system was commendable, exhibiting a high level of accuracy and a comparatively minimal degree of loss. RMSprop is renowned for its capacity to effectively accommodate various learning rate schedules, potentially playing a role in the commendable performance seen by this particular model.