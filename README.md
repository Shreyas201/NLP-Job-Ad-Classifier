# Job Advertisement Classifier

### Overview

This project aims to implement an automated job classification system leveraging text analysis advancements. By accurately predicting job categories for newly entered advertisements, it reduces human data entry errors, enhances job exposure to relevant candidates, and improves the overall user experience of the job hunting site.

### File Contents

- `data.zip` contains the job descriptions.
- `task1.ipynb` contains the code used to pre-process the job description such as tokenization, lowercase conversion, removing stop words, etc...
- `vocab.txt` contains the unigram vocabulary.
- `task2_3.ipynb` contains the code used to generate feature representations for job descriptions and then build ML models to classify job ads.
- `count_vectors.txt` contains sparse count vector representation of job advertisement. It starts with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line is the sparse representation of the corresponding description in the form of word_integer_index:word_freq separated by comma.
- `app.py` contains the code used to develop a web app using Python Flask. Where `static` and `templates` folders contain files necessary to design the front-end.
- `Advertisew2v_LR.pkl` is the classification model previously developed and is integrated into the app and `tkAdvertise_w2v.model` is the word embedding model.


