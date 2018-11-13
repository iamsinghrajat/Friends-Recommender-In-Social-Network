# LinkPrediction
Link Prediction in Social Netowork

Uses Networkx library for graph representation and score calculation

!pip install networkx in jupyter notebook to add this library

To run and understand this code  first reach ___EntryPoint___ (You can search for it in python code)

#facebook_combine.txt is dataset

#Facebook  dataset is included download others from http://snap.stanford.edu/data/ Use one with format like twitter_combine.txt or gplus_combine.txt

#At entry point you can choose which dataset to use you can even add your own dataset from http://snap.stanford.edu/data/ . Code will work for all dataset.

#Fn: Name of data set you want to run this code for, and cn is a integer for that dataset(any integer will work but different for each dataset)

#By default it is set to Facebook Data Set

#The project was run using Facebook and Twitter dataset but it works with any social network dataset from http://snap.stanford.edu/data/

#Following Scoring Methods are used to construct feature Set------------------


common_neighbors

resource_allocation_index

jaccard_coefficient

adamic_adar_index

preferential_attachment


#SVM ANN and Logistic Regresssion is used for classificaion
