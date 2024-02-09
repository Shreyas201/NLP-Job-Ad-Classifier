# Job Advertisement Classifier

### Overview

Part I: The objective of this project is to automate the classification of job advertisements using Natural Language Processing (NLP) techniques and machine learning models. The system preprocesses job ad text, generates feature representations, and applies machine learning models to predict job categories such as 'software development', 'marketing', and 'finance'. Due to the data being stored in multiple files and folders, a pipeline will be created to preprocess the data and automate the machine-learning steps.

Part II: We will develop a job-hunting website using the Python Flask web development framework. This website will allow job hunters to browse job advertisements and enable employers to create new job listings. Additionally, the website will incorporate a machine learning model from the first milestone to auto-classify job advertisement categories. This feature aims to enhance user experience and reduce data entry errors.

### File Contents

- `data.zip` contains the job descriptions.
- `task1.ipynb` contains the code used to pre-process the job description such as tokenization, lowercase conversion, removing stop words, etc...
- `task2_3.ipynb` contains the code used to generate feature representations for job descriptions and then build ML models to classify job ads.
- `app.py` contains the code used to develop a web app using Python Flask. Where `static` and `templates` folders contain files necessary to design the front-end.
- `Advertisew2v_LR.pkl` is the classification model previously developed and is integrated into the app and `tkAdvertise_w2v.model` is the word embedding model. 

