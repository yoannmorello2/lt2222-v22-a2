{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "04fa4397",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


1.
After opening with gzip.open, I used the function random.sample, which returns a random sampling without replacement.

2.
list_df_POStag(s) takes in entry the sampled list of sentences, strips them from the punctuation and the meaningless signs(like \n, or b\) in the beginning and the end. Then it returns a list of dataframes, with one df for each sentence in entry: one columns are the words, the other - their POS tag.

The output is fed to the next function: processed_df(list_of_df), which lower all the characters, gets rid of the non-alphabetic words and of the stop words.

create_samples(processed_sentences, samples) first drops the sentences with less than 5 words, then picks randomly (without replacement) n=samples of the left sentences. Finally, for each df chosen, picks a random 5 elements window.
It returns a list of df each with five rows corresponding to the selected five words window and their tags.

3.
transform(all_samples) transforms each df in the following array: first element= last two letters of the first word
                                                    #second elt = last two letters of the second word
                                                    #third elt= last two letters of the fourth word
                                                    #fourth elt = last two letters of the 5th word
                                                    #fifth element = POS of the third word
                                                    
create_df(all_samples) join all the df in the list into one big dataframe. Each row corresponds to a 5 words window. The first column are the last 2 letters of the first word, next, the last two letters of the second word, the last two letters of the fourth word, the last two letters of the 5th word and the POS of the third word.

The last column is transformed to contain one if the tag of the third word is 'NN' and 0 otherwise.
The four other columns are processed through OneHotEncoder, which does exactly the job that we need: to transform the unique values in each column into columns name with a one for the rows that contained this value and 0 otherwise.

4. 
We use sklearn train_test_split

5.
6.
From sklearn precision_recall_fscore_support we get a list of arrays. The firs three arrays gives the precision, the recall and the F_score Each array contains two elements: the first for the class 0 and the second for the class 1.
We chosed to predict the class 'NN' because it was from far the most common. Nevertheless it represents only  about 30% of the rows, so that a trivial learner which would predict 0 for any sample would have for the class 0 a precision of .7 and a recall of 1 (while having 0 of precision and recall for the class 1)
                   
mc.eval_model(model_rbf, test_X, test_y)
(array([0.72556126, 0.79104478]),
 array([0.99806121, 0.01907161]),
 array([0.84027049, 0.03724526]))
 
mc.eval_model(model_linear, test_X, test_y)
(array([0.7238269 , 0.40601504]),
 array([0.98905969, 0.01943145]),
 array([0.83590824, 0.03708791]))
 
At first glance, one can see that for everything else equal, the SVM with a rbf kernel has a higher precision than the linear one for label 1. In the cases when it predicts a 'NN', it is right 79% of the time (so there are some cases that it is quite good at predicting). But the recall for label 1 shows that both model are missing nearly all the 1.
If we compare both to the trivial learner predicting only 0s, we can see that the F-scores as well as the recall and the precision for category 0 are nearly the same:
 
Predicting all 0, we would get:
(array([0.7221, 0.    ]),
 array([1., 0.]),
 array([0.83862726, 0.        ]),
 
This is not surprising when we observe that the number of 1 in y_test is 2779, while the number of 1 in the prediction of the SVM_rbf is 67. Our best model predicts nearly always 0, except in some 67 cases where it seems it caught some pattern and make good predictions (80% precision).

Bonus question:
We tried a Naive Bayesian learner. Suprisingly, it predicts nearly all ones, resulting in F_scores worse than a trivial learner

 
