Link Prediction in Social Netowork
=================================

:info: A perfect repo for your  college project on data mining - link prediction - friends recommender.

How To Use
----------
#. 


    .. code-block:: shell

        pip install networkx
         
#.
 
    To run and understand this code  first reach ___EntryPoint___ in `python code <https://github.com/iamsinghrajat/Friends-Recommender-In-Social-Network/blob/9c0f4516123c8a5dd3163718704b375ab1c2c7da/LinkPredictionInSocialNetwork.py#L264>`_ . 

 
#.

    * facebook_combine.txt is dataset.

    * Facebook dataset is included download others from http://snap.stanford.edu/data/ . Use one with format like twitter_combine.txt or gplus_combine.txt .
    
    * At entry point you can choose which dataset to use you can even add your own dataset from http://snap.stanford.edu/data/ . Code will work for all dataset.
 
    *  By default it is set to Facebook Data Set

#. 
    * **sample_positive.txt:** all positive friend relation ie. people with connecting edge in graph
    * **sample_negative.txt:** all negative friend relation ie. people with no connecting edge in graph
    * **training.txt:** combination of positive and negative to get a dataset of relations
    * **features_combined_2.txt:** for each relation features like AdamicAdar are calculated and then used for training model


    

#.
    Following Scoring Methods are used to construct feature Set


    * common_neighbors

    * resource_allocation_index

    * jaccard_coefficient

    * adamic_adar_index

    * preferential_attachment


#.
    SVM ANN and Logistic Regresssion is used for classificaion
