import tkinter as tk
import tkinter as ttk
from ttkthemes import ThemedTk
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Smishing Detection Tool")

        self.title_label = tk.Label(self, text="Smishing Detector", font=("Helvetica", 24, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=4, pady=(20,0), padx=15, sticky=tk.NSEW)

        self.description_label = tk.Label(self, text="This tool analyses text messages to detect potential\nSMS phishing attempts.", font=("Helvetica", 22))
        self.description_label.grid(row=1, column=0, columnspan=4, pady=(0,5), padx=15, sticky=tk.NSEW)

        self.examples_label = tk.Label(self, text="Examples:", font=("Helvetica", 22))
        self.examples_label.grid(row=2, column=0, padx=20, sticky=tk.W)

        def insert_spam(event):
            text = "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed $1000 cash or $5000 prize!"
            self.entry.delete("1.0", tk.END)  
            self.entry.insert(tk.END, text)
        
        def insert_ham(event):
            text = "Hi ,where are you? We're at Jane's place and they're not keen to go out. I kind of am but feel I shouldn't so can we go out tomorrow? Don't mind do you?"
            self.entry.delete("1.0", tk.END) 
            self.entry.insert(tk.END, text)

        self.smishing_label = tk.Label(self, text="smishing", font=("Helvetica", 22), foreground='#3477eb')
        self.smishing_label.grid(row=3, column=0, pady=(0,10), padx=(20,10), sticky=tk.W)
        self.smishing_label.bind("<Enter>", lambda event: self.smishing_label.config(cursor="hand2"))
        self.smishing_label.bind("<Button-1>", insert_spam)

        self.nonsmishing_label = tk.Label(self, text="non-smishing", font=("Helvetica", 22), foreground='#3477eb')
        self.nonsmishing_label.grid(row=3, column=0, pady=(0,10), sticky=tk.E)
        self.nonsmishing_label.bind("<Enter>", lambda event: self.nonsmishing_label.config(cursor="hand2"))
        self.nonsmishing_label.bind("<Button-1>", insert_ham)
        
        self.entry = tk.Text(self, height=10, width=40, wrap="word", highlightcolor='#3477eb', highlightthickness=1, font=("Helvetica", 22))
        self.entry.grid(row=4, column=0,columnspan=4, padx=20, sticky=tk.NSEW)
        
        self.button = tk.Button(self, text="Verify", height=2, width=15, command=lambda: self.verifyMsg(), font=("Helvetica", 22))
        self.button.grid(row=5, column=0, columnspan=4, pady=28)

    def verifyMsg(self):
        countChar = lambda l1,l2: sum([1 for x in l1 if x in l2])
        countCapital = lambda l1: sum([1 for x in l1 if x.isupper()])
        countLower = lambda l1: sum([1 for x in l1 if x.islower()])
        countDigit = lambda l1: sum([1 for x in l1 if x.isdigit()])
        countKeyWord = lambda l1, l2: sum([1 for x in l1 if x.lower() == l2.lower()])
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        entry = str(self.entry.get("1.0", tk.END))
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        message = {'message': [entry]}
        input = pd.DataFrame(message)
        input['message_cleaned'] = input['message'].apply(self.preprocess_text)
        input['message_length'] = input['message'].apply(len)
        input['count_punctuations'] = input['message'].apply(lambda s: countChar(s, string.punctuation))
        input['count_uppercase'] = input['message'].apply(lambda s: countCapital(s))
        input['count_lowercase'] = input['message'].apply(lambda s: countLower(s))
        input['count_numerical_digits'] = input['message'].apply(lambda s: countDigit(s))
        input['contains_discount'] = input['message'].str.contains('discount', case=False).astype(int)
        input['contains_free'] = input['message'].str.contains('free', case=False).astype(int)
        input['contains_please'] = input['message'].str.contains('please', case=False).astype(int)
        input['contains_account'] = input['message'].str.contains('account', case=False).astype(int)
        input['contains_customer'] = input['message'].str.contains('customer', case=False).astype(int)
        input['contains_card'] = input['message'].str.contains('card', case=False).astype(int)
        input['contains_email'] = input['message'].str.contains('email', case=False).astype(int)
        input['contains_details'] = input['message'].str.contains('details', case=False).astype(int)
        input['contains_update'] = input['message'].str.contains('update', case=False).astype(int)
        input['contains_online'] = input['message'].str.contains('online', case=False).astype(int)
        input['contains_bank'] = input['message'].str.contains('bank', case=False).astype(int)
        input['contains_link'] = input['message'].str.contains('link', case=False).astype(int)
        input['contains_refund'] = input['message'].str.contains('refund', case=False).astype(int)
        input['contains_due'] = input['message'].str.contains('due', case=False).astype(int)
        tfidf_features = tfidf.transform(input['message_cleaned']).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf.get_feature_names_out())
        input = pd.concat([input, tfidf_df], axis=1)
        input.drop(['message','message_cleaned'], axis=1, inplace=True)
        predictions = model.predict(input)
        self.button.destroy()
        del self.button
        if predictions == 0:
            self.detection_label = tk.Label(self, text="No smishing content detected.\nThe message appears to be safe", font=("Helvetica", 22), foreground="#09870e")
            self.detection_label.grid(row=5, column=0, columnspan=4, pady=20, sticky=tk.NSEW)
        else:
            self.detection_label = tk.Label(self, text="Potential smishing content detect! Exercise\ncaution and do not provide personal information.", font=("Helvetica", 22), foreground="#a10c0a")
            self.detection_label.grid(row=5, column=0, columnspan=4, pady=20, sticky=tk.NSEW)
        self.empty_label = tk.Label(self, text='clear', font=('Helvetica', 22, 'bold'), background="white", foreground='#3477eb')
        self.empty_label.place(relx=0.9, rely=0.78, anchor=tk.CENTER)
        self.empty_label.bind("<Enter>", lambda event: self.empty_label.config(cursor="hand2"))
        self.empty_label.bind("<Button-1>", self.reset)

    def reset(self, event):
        self.button = tk.Button(self, text="Verify", height=2, width=15, command=lambda: self.verifyMsg(), font=("Helvetica", 22))
        self.button.grid(row=5, column=0, columnspan=4, pady=28)
        self.empty_label.destroy()
        del self.empty_label
        self.detection_label.destroy()
        del self.detection_label
        self.entry.delete("1.0", tk.END)
            
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        text = [word for word in text.split(' ') if word not in stop_words]
        return ' '.join(text)

    def stemming(self, text):
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text.split(' ')]
        return ' '.join(text)

    def preprocess_text(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stemming(text)
        return text
        
if __name__ == '__main__':
    app = App()
    app.mainloop()