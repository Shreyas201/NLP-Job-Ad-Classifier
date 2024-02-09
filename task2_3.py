#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Shreyas Ainapur
# #### Student ID: s3928704
# 
# Date: 25/09/2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# ## Introduction
# - Firstly all the libraries that are to be used in this notebook is loaded
# - Task 2 and Task 3 are addressed in this file
# - In Task 2 three vector representations are generated. `Count Vectors` from bag of word model and `Unweighted Word2Vec` and `Weighted Word2Vec` from bag of embedding model
# - `Weighted Word2Vec` is carried out using `tf-idf` features
# - At the end of the Task 2 `count_vectors.txt` file is generated
# - Task 3 is about building classification models
# - Firstly, classification models are build on the vector representations that are generated in Task 2
# - Followed by, three experiments are conducted to with the aim of building more robust model
# - classification model using only job title, classification model using only job description, and classification model using both job title and job description

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import numpy as np
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from gensim import utils
import gensim.models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.datasets import load_files


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# In[2]:


## loading the pre processed file performed in the Task 1
job_advertisementsFile = './job_advertisements.txt'
with open(job_advertisementsFile) as f: 
    tk_advertise = f.read().splitlines()
    
print(len(tk_advertise))
print(tk_advertise[11])


# In[3]:


## loading the vocab file created in the Task 1
## vocab is vocabulary file containing all the distinct words present in the tk_advertise with each word asssigned an 
## index value

vocabFile = './vocab.txt'
with open(vocabFile) as f: 
    lst_vocab = f.read().splitlines()

vocab = [lst_vocab[i].split(":")[0] for i in range(0,len(lst_vocab))]
print(len(vocab))


# ### 2.1 Generating Count Vector Features

# In[4]:


# Code to perform the task...
cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(tk_advertise)
print(count_features.shape)
print(count_features[0])


# In[5]:


## vectorized form[using count vector] of tokenised data is represented in a data frame with columns as vocab
count_vect_df = pd.DataFrame(count_features.todense(), columns = vocab)
print(count_vect_df.shape)
print(count_vect_df.isna().sum().sum())
count_vect_df.head()


# In[6]:


## creating a list that stores sparse count vector representation from the obtained "count_features" in the above step
count_vector_lst = []
num = count_features.shape[0] # the number of document
for a_ind in range(0, num): # loop through each article by index
    count_vector_dict = {}
    for f_ind in count_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
        value = count_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
        count_vector_dict[str(f_ind)] = value
    count_vector_lst.append(count_vector_dict)

print(len(count_vector_lst))


# ### 2.2 Using Word2Vec pre-trained model to generate unweighted Vector Features

# In[7]:


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, corpusFile):
        MyCorpus.fPath = corpusFile # specific the path to the corpus file
    def __iter__(self):
        for line in open(self.fPath):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


# In[8]:


## crating a word embbeding model using "Word2Vec" model on the pre-processed text file "job_advertisements" from Task 1
sentences = MyCorpus('./job_advertisements.txt')
tkAdvertise_w2v = gensim.models.Word2Vec(sentences=sentences)


# In[9]:


## extracting keyedvetors from the model
tkAdvertisew2v_wv = tkAdvertise_w2v.wv
tkAdvertisew2v_wv['south']


# In[10]:


## defining a function for vector representation of documents
def gen_docVecs(wv,tk_txts): 
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    
    for i in range(0,len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors


# In[11]:


## tokenising pre-processed text
tk_advertise_words = [sent.split(" ") for sent in tk_advertise]


# In[12]:


## generating vector representation for documents 
preTW2v_tkAdvertise = gen_docVecs(tkAdvertisew2v_wv,tk_advertise_words)
print(preTW2v_tkAdvertise.shape)
print(preTW2v_tkAdvertise.isna().any().sum())
preTW2v_tkAdvertise


# In[13]:


## extracting rows with null values in the DataFrame "preTW2v_tkAdvertise"
rows_NaN = preTW2v_tkAdvertise[preTW2v_tkAdvertise[0].isnull()]
rows_NaN


# In[14]:


## storing the index posistions of the documents that is casuing the feature repesntation to have null values
missing_idx = list(rows_NaN.index)
missing_idx


# In[15]:


## total number of null values in the DataFrame
print("Total number of null values in the DataFrame 'preTW2v_tkAdvertise': "+str(preTW2v_tkAdvertise.isna().sum().sum()))

## finding the value in any two documents which is returned as a row with null values in the above DataFrame
print("Text in the pre-processed job_description file 'tk_advertise': "+tk_advertise[34])
print("Text in the pre-processed job_description file 'tk_advertise': "+tk_advertise[196])


# - after pre-processing the corpus there are few documents which are empty or repetative word
# - vector representation of these documents is returning null values
# - thus these cannot be imputed nor has any value providng to the DataFrame and is to be removed

# In[16]:


## remove rows with null values
preTW2v_tkAdvertise = preTW2v_tkAdvertise.dropna()
print(preTW2v_tkAdvertise.isna().sum().sum())
print(preTW2v_tkAdvertise.shape)


# - 14 documents had null values which are now removed

# ### 2.3 Using Word2Vec pre-trained model to generate weighted Vector Features

# In[17]:


## creating tf-idf features of each word in vocab w.r.t the documents
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform([' '.join(article) for article in tk_advertise_words]) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[18]:


print(tfidf_features[1])


# In[19]:


## saving the "tfidf_features" in text format
def write_vectorFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file
    
tVector_file = "./jobAdvertise_tVector.txt" # file name of the tfidf vector
write_vectorFile(tfidf_features,tVector_file) # write the tfidf vector to file


# In[20]:


## defining a function to read the vocab text file and create a dictionary in the format "index:word"
def gen_vocIndex(voc_fname):
    with open(voc_fname) as vocf: 
        voc_Ind = [l.split(':') for l in vocf.read().splitlines()] # each line is 'index,word'
    return {int(vi[1]):vi[0] for vi in voc_Ind}


# Generates the w_index:word dictionary
voc_fname = './vocab.txt' # path for the vocabulary
voc_dict = gen_vocIndex(voc_fname)
print(len(voc_dict))


# In[21]:


def doc_wordweights(fName_tVectors, voc_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents
    num_idx =[] # a list containing idex positions of documents that are empty
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector   
        try:
            tv = tv.strip()
            weights = tv.split(' ') # list of 'word_index:weight' entries
            weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
            wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
            tfidf_weights.append(wordweight_dict)
        except:
            tfidf_weights.append({'NaN':'NaN'})
        
    return tfidf_weights

fName_tVectors = 'jobAdvertise_tVector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)
print(len(tfidf_weights))
tfidf_weights


# In[22]:


## defining a function to create vector representation representation of words in each document using wieghted method
def gen_docVecs(wv,tk_txts,tfidf = []): # generate vector representation for documents
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    #stopwords = nltk.corpus.stopwords.words('english') # removing stop words

    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # get the list of distinct words of the document

        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in embeddings then proceed

                if tfidf != []:
                    word_weight = float(tfidf[i][str(word)])
                else:
                    word_weight = 1
                temp = temp.append(pd.Series(word_vec*word_weight), ignore_index = True) # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum() # take the sum of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors

weighted_preTW2v_dvs = gen_docVecs(tkAdvertisew2v_wv,tk_advertise_words,tfidf_weights)
print(weighted_preTW2v_dvs.shape)
print(weighted_preTW2v_dvs.isna().any().sum())
weighted_preTW2v_dvs


# In[23]:


rows_NaN = weighted_preTW2v_dvs[weighted_preTW2v_dvs[0].isna()]
rows_NaN


# In[24]:


## storing the index posistions of the documents that is casuing the feature repesntation to have null values
missing_idx = list(rows_NaN.index)
missing_idx


# In[25]:


## total number of null values in the DataFrame
print("Total number of null values in the DataFrame 'weighted_preTW2v_dvs': "+str(weighted_preTW2v_dvs.isna().sum().sum()))

## finding the value in any two documents which is returned as a row with null values in the above DataFrame
print("Text in the pre-processed job_description file 'tk_advertise': "+tk_advertise[34])
print("Text in the pre-processed job_description file 'tk_advertise': "+tk_advertise[196])


# - after pre-processing the corpus there are few documents which are empty or repetative word
# - vector representation of these documents is returning null values
# - thus these cannot be imputed nor has any value providng to the DataFrame and is to be removed

# In[26]:


## remove rows with null values
weighted_preTW2v_dvs = weighted_preTW2v_dvs.dropna()
print(weighted_preTW2v_dvs.isna().sum().sum())
print(weighted_preTW2v_dvs.shape)


# - 14 documents had null values which are now removed

# ### 2.4 Generating "count_vectors" text file

# In[27]:


## loading the text documents
files = load_files(r"data")
print(len(files['target']))
print(files.target_names)


# In[28]:


## defining a function for the tokenization of the text files
## extracting title, webindex in lists
def TitleWebindex(job_advertise):

    ## convert the bytes-like object to python string
    advertise = job_advertise.decode('utf-8')     
    
    ## segament into sentences
    sentences = sent_tokenize(advertise)
    
    ## tokenize each sentence
    pattern = r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?"
    
    ## using regex tokeniation is carried out
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    ## merging them into a list of tokens
    tokenised_advertise = list(chain.from_iterable(token_lists))
    
    ## finding the index position of description in the list and manipulating list[tokenised_advertise] such that only 
    ## description of job advertisement is extracted
    idx_title = tokenised_advertise.index('Title')
    idx_webindex = tokenised_advertise.index('Webindex')
    title = tokenised_advertise[idx_title+1]
    webindex = tokenised_advertise[idx_webindex+1]
        
    return title, webindex


# In[29]:


job_advertise = files.data
tk_title = [TitleWebindex(job)[0] for job in job_advertise]
tk_webindex = [TitleWebindex(job)[1] for job in job_advertise]
print(len(tk_title))
print(len(tk_webindex))


# - we have title, webindex, target_names in list
# - relation between target_names and target:-
# - 0 - 'Accounting_Finance'
# - 1 - 'Engineering'
# - 2 - 'Healthcare_Nursing'
# - 3 - 'Sales'

# In[30]:


## creating a data frame with following as the columns
df_corpus = pd.DataFrame()
df_corpus['Title'] = tk_title
df_corpus['Webindex'] = tk_webindex
df_corpus['tk_text'] = tk_advertise
df_corpus['tk_tokens'] = tk_advertise_words
df_corpus['Category'] = files.target
print(df_corpus.shape)
df_corpus.head()


# In[31]:


## saving the output
## creating text file "count_vectors" using vector representation done in 2.1
def write_countVector(df,lst,filename):
    num = df.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each article by index
        string = "#"+str(df['Webindex'][a_ind])
        for key in count_vector_lst[a_ind]:
            val = count_vector_lst[a_ind][key]
            string += ","+str(key)+":"+str(val)
        out_file.write(string) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each article
    out_file.close() # close the file
    
count_vectors = "./count_vectors.txt" 
write_countVector(df_corpus,count_vector_lst,count_vectors)


# ## Task 3. Job Advertisement Classification

# ### 3.1 Classification model buidling on the feature representations obtained in 2.1,2.2, and 2.3
# ### 3.1.1 Machine learning model on count vector features

# In[32]:


seed = 0 # set a seed to make sure the experiment is reproducible


# In[33]:


# creating training and test split
X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(count_vect_df, df_corpus['Category'], 
                                                              list(range(0,len(df_corpus))),test_size=0.33,
                                                              random_state=seed)

model_cv = LogisticRegression(max_iter = 2000,random_state=seed)
model_cv.fit(X_train, y_train)
model_cv.score(X_test, y_test)


# In[34]:


## defining a function to plot the graph to view the vector representation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def plotTSNE(labels,features): # features as a numpy array, each element of the array is the document embedding of an article
    categories = sorted(labels.unique())
    # Sampling a subset of our dataset because t-SNE is computationally expensive
    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for i in range(0,len(categories)):
        points = projected_features[(labels[indices] == categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=categories[i])
    plt.title("Feature vector for each article, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()


# In[35]:


features = count_vect_df.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(df_corpus['Category'],features) # plot the tSNE to have a look


# ### 3.1.2 Machine learning model on unweighted Word2Vec model

# In[36]:


## documents at these index positions are removed
missing_idx


# In[37]:


## storing the labels[0,1,2,3] which are present in the "Category" column of DataFrame "df_corpus" and removing the ones
## occuring the the same index positions as in "missing_idx"
category = [df_corpus['Category'][i] for i in range(0,len(df_corpus)) if i not in missing_idx]
category = pd.Series(category)
print(len(category))


# In[38]:


# creating training and test split
X_train, X_test, y_train, y_test = train_test_split(preTW2v_tkAdvertise, category, test_size=0.2,random_state=seed)
                                                                                                                          
model_unwt = LogisticRegression(max_iter = 1000,random_state=seed)
model_unwt.fit(X_train, y_train)
model_unwt.score(X_test, y_test)


# In[39]:


features = preTW2v_tkAdvertise.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(category,features) # plot the tSNE to have a look


# ### 3.1.3 Machine learning model on weighted[tf_idf] Word2Vec model

# In[40]:


# creating training and test split
X_train, X_test, y_train, y_test = train_test_split(weighted_preTW2v_dvs, category, test_size=0.2,random_state=seed)
                                                                                                                          
model_wt = LogisticRegression(max_iter = 1000,random_state=seed)
model_wt.fit(X_train, y_train)
model_wt.score(X_test, y_test)


# In[41]:


features = weighted_preTW2v_dvs.to_numpy() # convert the document vector dataframe to a numpy array
plotTSNE(category,features) # plot the tSNE to have a look


# In[42]:


from sklearn.model_selection import KFold
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[43]:


def evaluate(X_train,X_test,y_train, y_test,seed=0):
    model = LogisticRegression(random_state=seed,max_iter = 1000)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[44]:


cv_df = []
emb_df = []
wgt_df = []

fold = 0
while fold < 5:
    
    X_train, X_test, y_train, y_test = train_test_split(count_vect_df, df_corpus['Category'],test_size=0.2,random_state=seed)                                                        
    ans = evaluate( X_train, X_test, y_train, y_test,seed)
    cv_df.append(ans)

    X_train, X_test, y_train, y_test = train_test_split(preTW2v_tkAdvertise, category, test_size=0.2,random_state=seed)
    ans = evaluate( X_train, X_test, y_train, y_test,seed)
    emb_df.append(str(ans))
    
    X_train, X_test, y_train, y_test = train_test_split(weighted_preTW2v_dvs, category, test_size=0.2,random_state=seed)
    ans = evaluate( X_train, X_test, y_train, y_test,seed)
    wgt_df.append(str(ans))
    
    fold +=1
    
## creating a DataFrame to store accuracy of each model on each iteration
acc_df = pd.DataFrame()
acc_df['CountVector'] = cv_df
acc_df['Unweighted_Word2Vec'] = emb_df
acc_df['Weighted_Word2Vec'] = wgt_df
acc_df


# ### 3.2 Classification Models

# ### 3.2.1 Classification model with only job title

# In[45]:


## job title is stored in list "tk_title"
tk_title


# In[46]:


tk_title = [title.lower() for title in tk_title]


# In[47]:


## creating a DataFrame to store job title and classes of the job advertisements under category column
df_model_1 = pd.DataFrame()
df_model_1['Title'] = tk_title
df_model_1['Category'] = files.target
df_model_1.head()


# In[48]:


## generating a vector representation of job title using "Count Vectors" 
CV_Title = cVectorizer.fit_transform(tk_title)


# In[49]:


# creating training and test split
X_train, X_test, y_train, y_test = train_test_split(CV_Title, files.target, test_size=0.2, random_state=seed)                                                        

## building classification model using "LogisticRegression"
model_title = LogisticRegression(max_iter = 1000,random_state=seed)
model_title.fit(X_train, y_train)

## predicting the model accuracy
model_title.score(X_test, y_test)


# In[50]:


## initialising 5 fold validation
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True)
print(kf)


# In[51]:


## evaluating the model using 5-fold cross validation method
cv_title = []
fold = 0
for train_index, test_index in kf.split(list(range(0,len(files.target)))):
    y_train = [str(files.target[i]) for i in train_index]
    y_test = [str(files.target[i]) for i in test_index]

    X_train_count, X_test_count = CV_Title[train_index], CV_Title[test_index]
    ans = evaluate(CV_Title[train_index],CV_Title[test_index],y_train,y_test,seed)
    print("Accuarcy of the classification model using only job title for iteration "+str(fold)+" is "+str(ans))
    cv_title.append(ans)
    
    fold +=1
avg_acc = sum(cv_title)/len(cv_title)
print("Average accuracy of the model achieved on a 5-fold cross validation is "+str(avg_acc))


# - the average accuracy of the classification model when considered only the job title is `51.68%`
# - the relation between model accuracy and iteration is non-linear
# - it first increases then decreases and increases and agian decreases
# - highest accuracy is achieved on the fourth iteration of the 5-fold cross validation which is `56.13%`

# ### 3.2.2 Classification model with only job description

# - the model_cv is already generated in section `3.1.1`
# - `model_cv` is the logistic regeression model when only job description is considered
# - `model_cv` is generated using `Count Vectors`
# - below is the 5-fold cross validation

# In[52]:


## evaluating the model using 5-fold cross validation method
cv_desc = []
fold = 0
for train_index, test_index in kf.split(list(range(0,len(files.target)))):
    y_train = [str(files.target[i]) for i in train_index]
    y_test = [str(files.target[i]) for i in test_index]

    X_train_count, X_test_count = count_features[train_index], count_features[test_index]
    ans = evaluate(count_features[train_index],count_features[test_index],y_train,y_test,seed)
    print("Accuarcy of the classification model using only job description for iteration "+str(fold)+" is "+str(ans))
    cv_desc.append(ans)
    
    fold +=1
avg_acc = sum(cv_desc)/len(cv_desc)
print("Average accuracy of the model achieved on a 5-fold cross validation is "+str(avg_acc))


# - the average accuracy of the classification model when considered only the job description is `75.90%`
# - the relation between model accuracy and iteration is non-linear
# - it first increases then decreases and increases
# - highest accuracy is achieved on the second iteration of the 5-fold cross validation which is `78.70%`

# ### 3.2.3 Classification model with both job description and job title

# In[53]:


## concatenating job description and job title 
desc_title = [tk_advertise[i]+" "+tk_title[i] for i in range(0,len(tk_advertise))]


# In[54]:


## generating vector representation using count vectors
cv_desc_title = cVectorizer.fit_transform(desc_title)


# In[55]:


# creating training and test split
X_train, X_test, y_train, y_test = train_test_split(cv_desc_title, files.target, test_size=0.2, random_state=seed)                                                        

## building classification model using "LogisticRegression"
model_desc_title = LogisticRegression(max_iter = 1000,random_state=seed)
model_desc_title.fit(X_train, y_train)

## predicting the model accuracy
model_desc_title.score(X_test, y_test)


# In[56]:


## initialising 5 fold validation for model evaluation
num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True)
print(kf)


# In[57]:


## evaluating the model using 5-fold cross validation method
acc_desc_title = []
fold = 0
for train_index, test_index in kf.split(list(range(0,len(files.target)))):
    y_train = [str(files.target[i]) for i in train_index]
    y_test = [str(files.target[i]) for i in test_index]

    X_train_count, X_test_count = cv_desc_title[train_index], cv_desc_title[test_index]
    ans = evaluate(cv_desc_title[train_index],cv_desc_title[test_index],y_train,y_test,seed)
    print("Accuarcy of the classification model using only job title for iteration "+str(fold)+" is "+str(ans))
    acc_desc_title.append(ans)
    
    fold +=1
avg_acc = sum(acc_desc_title)/len(acc_desc_title)
print("Average accuracy of the model achieved on a 5-fold cross validation is "+str(avg_acc))


# - the average accuracy of the classification model when considered only the job description is `77.06%`
# - the relation between model accuracy and iteration is non-linear
# - it first increases then decreases and increases
# - highest accuracy is achieved on the second iteration of the 5-fold cross validation which is `79.35%`

# ## Summary
# - In Task 2, `Count Vector` features are generated using the function `cVectorizer` and the vocabulary file `vocab`
# - It gives the number of times each word in the vocab file has arrived in the document
# - `Unweighted Word2Vec` is a model of word of embeding which first calculates vector representation of each word in a document. Then, summing up the these values gives the vector representation of that document
# - When this model is converted into DataFrame the columns are the dimention of the vector. Here it is 100
# - `Weighted Word2Vec` is also similar to `Unweighted Word2Vec` but this model also considers the weightage of each word and is obtained using tf-idf vectorizer
# - In Task3, classification models on these three vector represntations are build
# - Highest accuracy of the model is achieved by the "count vector" model which is `78.21%`, followed by "Unweighted Word2Vec" model `71.24%` and lowest accuracy is obtained by "Weighted Word2Vec" model which is `40.52%`
# - Moreover, three experiments are conducted to build more robust model
# - `LogisticRegeression` is used for building classification model and `5-fold cross-validation` method is used for the evaluation
# - Firstly, classification model is built using only job title however, the model has very low accuracy. Even after performing 5-fold cross validation the highest acuracy is as high as `56.13%`
# - Then, classification model is built using only job description. In this case highest accuracy achieved is `78.70%`
# - Finally, classification model is built using both job title and job description and the highest accuracy obtained by the model is `79.53%`
# #### After conducting these experiments, it is concluded that it is possible to increase model performance after including job title however, the increase in performance is not very significant. Thus, the feature job title is not a significant feature for the classification model
