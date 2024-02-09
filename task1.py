#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Shreyas Ainapur
# #### Student ID: s3928704
# 
# Date: 22/09/2022
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# - In this Task 1 pre-processing of the job escription from the given data set is carried out
# - Firstly, the given data is examined
# - Then, different pre-processing functions are applied the data such as, tokenization, lower case conversion, removing stop words, etc...
# - After cleaning the data, three different files are saved in the text format in the local directory:
# - `job_advertisements.txt` contains pre-processed data
# - `sentiments.txt` contains the labels[category of the job]
# - `vocab` contains the vocabulary of the cleaned job advertisement descriptions 

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.probability import *


# ### 1.1 Examining and loading data
# - xamine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 
# 
# - In the `data` folder there are 4 more sub folders `Accounting_Finance`, `Engineering`, `Healthcare_Nursing`, and `Sales`. - Each of these folders contains text files of job advertisement. This includes `Title`, `Webindex`, `Comapnay`, and `Description` of the job. Note that each file has 1 job details.
# - Accounting_Finance: 191 jobs
# - Engineering: 231 jobs
# - Healthcare_Nursing: 198 jobs
# - Sales: 156 jobs
# - `Data` folder also contains a text document that has all the `stop words`.
# - Finally, it contains two jupyter notebook files that has the framework of `task1` and `task2_3` respectively.

# In[2]:


# Code to inspect the provided data file...
files = load_files(r"data")


# In[3]:


files


# In[4]:


files['target']


# In[5]:


files['target_names']


# ### 1.2 Pre-processing data
# ### 1.2.1 Tokenization

# In[6]:


## defining a function for the tokenization of the text files
def tokenizeAdvertise(job_advertise):

    ## convert the bytes-like object to python string
    advertise = job_advertise.decode('utf-8')     
    
    ## segament into sentences
    sentences = sent_tokenize(advertise)
    
    ## tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    
    ## using regex tokeniation is carried out
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    ## merging them into a list of tokens
    tokenised_advertise = list(chain.from_iterable(token_lists))
    
    ## finding the index position of description in the list and manipulating list[tokenised_advertise] such that only 
    ## description of job advertisement is extracted
    idx = tokenised_advertise.index('Description')
    tokenised_advertise = tokenised_advertise[idx+1:]
        
    return tokenised_advertise


# In[7]:


## defining a function to get the stats of the tokenized description of job advertisements
def stats_print(tk_advertise):
    
    ## we put all the tokens in the corpus in a single list
    words = list(chain.from_iterable(tk_advertise)) 
    
    ## compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    vocab = set(words)
    
    ## get the stats
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of advertises:", len(tk_advertise))
    
    lens = [len(article) for article in tk_advertise]
    print("Average advertise length:", np.mean(lens))
    print("Maximun advertise length:", np.max(lens))
    print("Minimun advertise length:", np.min(lens))
    print("Standard deviation of advertise length:", np.std(lens))


# In[8]:


job_advertise = files.data
tk_advertise = [tokenizeAdvertise(job) for job in job_advertise]


# In[9]:


## comparing the raw text and tokenised text
print("Raw text:\n",job_advertise[11],'\n')
print("Tokenized review:\n",tk_advertise[11])


# In[10]:


stats_print(tk_advertise)


# ### 1.2.2 Lower Case Conversion

# In[11]:


## since tk_advertise has list with in list, first we loop to access each loop contents and then iterated through all words
## in each list to convert it to lower case
for i in range(0,len(tk_advertise)):
    tk_advertise[i] = list(map(lambda x: x.lower(), tk_advertise[i]))


# In[12]:


## checking for the conversion
tk_advertise[11]


# ### 1.2.3 Remove words with length less than 2

# In[13]:


## since tk_advertise has list with in list, first we loop to access each loop contents and then iterated through all words
## in each list to remove words with length less than 2
#for i in range(0,len(tk_advertise)):
#    [tk_advertise[i].remove(word) for word in tk_advertise[i] if len(word)<2]

tk_advertise = [[w for w in advertise if len(w) >=2]                       for advertise in tk_advertise]


# In[14]:


tk_advertise[11]


# ### 1.2.4 Removing stopwords

# In[15]:


stopword_df = pd.read_csv("stopwords_en.txt", header = None)

stopwords = list(stopword_df[0])
stopwords


# In[16]:


## since tk_advertise has list with in list, first we loop to access each loop contents and then iterate through all words
## in each list to remove stop words
#for i in range(0,len(tk_advertise)):
#    [tk_advertise[i].remove(word) for word in tk_advertise[i] if word in stopwords]
tk_advertise = [[w for w in advertise if w not in stopwords]                       for advertise in tk_advertise]


# In[17]:


sorted(tk_advertise[11])


# In[18]:


## checking the change in stats
stats_print(tk_advertise)


# ### 1.2.5 Remove the word that appears only once in the document collection, based on term frequency

# In[19]:


term_fd = [FreqDist(tk_advertise[i]) for i in range(0,len(tk_advertise))]


# In[20]:


term_fd


# In[21]:


len(term_fd[11])


# In[22]:


lessFreqWords = [list(set(term_fd[i].hapaxes())) for i in range(0,len(term_fd))]
lessFreqWords


# In[23]:


sorted(lessFreqWords[11])


# In[24]:


len(lessFreqWords)


# In[25]:


## removing words that appear only once in the document
tokenised_words = []
for i in range(0,len(tk_advertise)):
    ans = []
    removed_ans = []
    for w in tk_advertise[i]:
        if w in lessFreqWords[i]:
            pass
        else:
            ans.append(w)
    tokenised_words.append(ans)

tokenised_words


# In[26]:


tokenised_words[11]


# In[27]:


print(len((tk_advertise[11])))
print(len((lessFreqWords[11])))
print(len((tokenised_words[11])))


# In[28]:


stats_print(tokenised_words)


# ### 1.2.6 Remove the top 50 most frequent words based on document frequency

# In[29]:


## values of tk_advertise are a list containing each document's tokenised words
## lst_set is created such that duplicate words are removed from each list of tk_advertise
lst_set = []
for i in range(0,len(tokenised_words)):
    ans = [words for words in tokenised_words[i]]
    unq_ans = list(set(ans))
    lst_set.append(unq_ans)


# In[30]:


## checking length it should be same as the number of total documents
len(lst_set)


# In[31]:


## computing document frequency for each unique word
words = list(chain.from_iterable(lst_set))
doc_fd = FreqDist(words)


# In[32]:


## creating list top 50 words that are most frequent
top_50_words = []
for i in range(0,50):
    top_50_words.append(doc_fd.most_common(50)[i][0])


# In[33]:


## removing top 50 most frequent words based on document frequency
tk_advertise = [[w for w in advertise if w not in top_50_words]                       for advertise in tokenised_words]


# In[34]:


sorted(tk_advertise[11])


# In[35]:


## checking the change in stats
stats_print(tk_advertise)


# ### 1.2.7 Saving job advertisements and information

# In[36]:


sentiments = files.target


# In[37]:


def save_advertise(advertiseFilename,tk_advertise):
    out_file = open(advertiseFilename, 'w') # creates a txt file and open to save the reviews
    string = "\n".join([" ".join(advertise) for advertise in tk_advertise])
    out_file.write(string)
    out_file.close() # close the file
    
def save_sentiments(sentimentFilename,sentiments):
    out_file = open(sentimentFilename, 'w') # creates a txt file and open to save sentiments
    string = "\n".join([str(s) for s in sentiments])
    out_file.write(string)
    out_file.close() # close the file 


# In[38]:


save_advertise('job_advertisements.txt',tk_advertise)


# In[39]:


save_sentiments('sentiments.txt',sentiments)


# ### 1.2.8 Building a vocabulary of the cleaned job advertisement descriptions

# In[40]:


## adding all the words in a single list from the pre-processed tokenised list of words which is stored in tk_advertise
words_vocab = list(chain.from_iterable([w for w in tk_advertise]))

## dictionary to store index for each word
vocab_dict = {}
i = 0

## removing the duplicates
vocab_sorted = sorted(list(set(words_vocab)))

## word_set contains the distinct words occuring in job description documents
for word in vocab_sorted:
    vocab_dict[word] = i
    i += 1


# In[41]:


vocab_dict


# In[42]:


def save_vocab(vocabFilename,vocab_dict):
    out_file = open(vocabFilename, 'w') # creates a txt file and open to save the reviews
    string = "\n".join(["".join(key+":"+str(value)) for (key,value) in vocab_dict.items()])
    out_file.write(string)
    out_file.close() # close the file


# In[43]:


save_vocab('vocab.txt', vocab_dict)


# ## Summary
# - After cleaning only job description data, the final vocabulary size is `2767`
# - It also implies that the length of the `vocab` file is 2767
# - The total number of documents are `776`
# - The total number of tokens in the cleaned job description data is `27735`
