# Smishing Detector
This repository contains code related to the proposal and implementation of a solution for detecting SMS phishing (smishing) attacks using Natural Language Processing (NLP) methods.

The structure of the repository is as follows:
* code/: contains the python code for the development of the smishing detector
  * gui.py: implements the GUI for the application
  * model.py: implements code for training and evaluating seven classifiers
  * model.pkl: the nominated pretrained machine learning model for smishing detection
  * tfidf.pkl: fitted TF-IDF vectoriser for text vectorisation
* data/: contains the data used for the dataset
  * spam.csv: CSV file containing labeled SMS messages

### Usage
1. Run the app.py file to launch the GUI for the Smishing Detector.
2. Input a text message into the text box or use the provided examples.
3. Click the "Verify" button to analyse the message.
4. The result (smishing or non-smishing) will be displayed.

> Alternatively, you can run the model.py file to view the performance of the machine learning models without the GUI. This will provide insights into the accuracy and other metrics of each model.
