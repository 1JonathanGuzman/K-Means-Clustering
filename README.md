# K-Means-Clustering
Programs created to perform K-Means clustering on a given set of data. This project is organized into two folders containing two sub-projects:

K-Means_C++: 

    This folder contains two files

        k-means.cpp - A .cpp file that parses the data in Mall_Customers.csv, performs several iterations of K-means clustering, then calculates sum of squared error based on the final cluster assignments.

        Mall_Customers.csv - A .csv file consisting of 200 records of shopping data of 200 customers. This data was found on Kaggle at the following page: https://www.kaggle.com/datasets/shwetabh123/mall-customers

    To run the project:

        This project was developed and tested on Linux using g++, and can be run on such without any further configuration. This project can be run on other IDEs such as virtual studio or operating systems like Windows 10/11, but is likely to require further
        configuration.

K-Means_Python:

    This folder contains two files

        k-means.py: A .py file that performs k-means clustering on the seeds.txt dataset, then returns the sum of squared error on several runs of the algorithm using different numbers of centroids.

        seeds.txt: A dummy dataset consisting of two-dimensional numerical data created for our use by my instructor at Florida State University for my CIS4930 (Special Topics in Computer Science: Data Mining) course.
                    The data in this file is delimited by spaces (to separate features) and lines (to separate records).

    To run the project:

        This project was created and tested on Python 3.12. It uses the following external libraries:
            - numpy (2.0.0rc2)
        It should be run using the specified version of Python or newer, in a virtual environment that has numpy installed. 
        For those using the PyCharm IDE, you can either install numpy using pip or by using the "Manage Packages" option.
