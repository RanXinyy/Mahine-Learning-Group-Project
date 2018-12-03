<font face = "Times New Roman" size="3">
#Machine Learning Project - Forecasting HIV Cases Based on Google Trend

This is a program to implement Support Vector Regression and Neural Networks using python and predict HIV cases in the future.


##GitHub Link
https://github.com/RanXinyy/Mahine-Learning-Group-Project   

##Dataset and Requirements
* Dataset:  
The data file **data_final.xlsx** contains 1344 search index and 56 HIV cases respectively. Each index of 24 keywords and a HIV case in a quarter are a line in a file to become a vector. Each keywords feature represents keywords which HIV infected people like to search and search index in google trend. The last feature(e.g.#label#:Y) in each line indicates the HIV cases in each quarter.

* Requirements:  
Python    3.6.5   
numpy    1.14.3   
pandas   0.23.0    
scikit-learn  0.19.1   
matplotlib  2.2.2   
scipy   1.1.0   
seaborn   0.8.1 

For your convenience, you can get all the installation materials through the link below:
(https://drive.google.com/drive/folders/1ivhN4XCDXCtAr9sr9hEPeQwDv_mvWPyc?usp=sharing)


###Running instruction：  
1. **GroupAssignment.py** and **data_final.xlsx** should be in the same folder.  
2. Printed results are the quantitive prediction effect in both Neural Network and SVR using different parameters and kernel functions. ‘rmse’ stands for Root Mean Square Error.  

##How to Run
* How to run Nerual Network?  
Pre-condition: Run all functions from Line 21 to Line 109.  
First, load data 'data_final.xls', split dataset and normalize variables. ----- Line 113 to 116.  
Second, select the best parameters-----Line 119 to 128.  
Third, set up nerual network model-----Line 130.  
Finally, plot ----- More are referred to Procedures to plot six kinds of figures below.  


* How to run SVR?  
Run the code from Line 208 to Line 328.

* Procedures to plot six kinds of figures:  

&emsp;&emsp;1. Plot of HIV Trends: Go to Line 192 and uncomment the annotated code from Line 192 to Line 205.  
&emsp;&emsp;2. Plot of Thermal map to review the correlation between search index of 24 keywords and HIV cases: Go to Line 149 and uncomment the annotated code from Line 149 to Line 160.  
&emsp;&emsp;3. Plot of correlation coefficient: Go to Line 329 and uncomment the annotated code from Line 329 to line 334.  
&emsp;&emsp;4. Plot of prediction effect in neural network: Go to Line 162 and uncomment the annotated code from Line 162 to Line 189.  
&emsp;&emsp;5. Plot of fluctation using 10 folds validatation: Go to Line 132 and uncomment the annotated code from Line 132 to Line 147.   
&emsp;&emsp;6. Plot printed is the prediction effect of SVR.
##Notice
In order to show our structure of project code more clearly, we upload **demo.ipynb**. And in our demo, we run it in jupyter NoteBook IDE. The content of demo.ipynb is the same as GroupAssignmnet.py, and demo.ipynb is only for demo recording.
</font>