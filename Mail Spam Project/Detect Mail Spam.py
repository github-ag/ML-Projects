#########################    Importing the Machine Learning libraries     ###################################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



############ Header files for Data Cleaning ###################
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

##########  Header file for vectorization   ##################
from sklearn.feature_extraction.text import CountVectorizer


###################     Header files for reading the mail from the id     ################################

import imaplib
import email


#############                              PREPARING THE DATA FOR TRAINING                                 ##############

with open("Sample Mails.txt",'r') as file:
	data = file.readlines()

X = []
Y = []

# divinding the data into label and message part
for mail in data:
	mail_parts = mail.split('\t')
	label = mail_parts[0]
	message = ' '.join(mail_parts[1:]) #joining the message parts while seperating ham/spam from the mail.
	X.append(message)
	Y.append(label)

	#print(len(X))
	#print(len(Y))


#####################    Cleaning the data    ##############
tokenizer = RegexpTokenizer(r'\w+')
sw = set(stopwords.words('english'))
ss = SnowballStemmer("english")

def clean_mail(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("\n","")
    word_list = tokenizer.tokenize(sentence)
    word_list = [w for w in word_list if w not in sw]
    final_list = [ss.stem(w) for w in word_list]
    new_sentence = ' '.join(final_list)
    
    return new_sentence

X_clean = [clean_mail(i) for i in X]

###############        Vectorizing the data to convert into numerical data    #####

cv = CountVectorizer(ngram_range=(1,2))
x_vec = cv.fit_transform(X_clean).toarray()
#print(x_vec)
#print(x_vec.shape)


######################    GETTING THE LAST EMAIL FROM THE ID     ########################

def get_last_message():
    user = "gotravel.agsr@gmail.com"
    password = "AbhishekSachin"
    imap_url = "imap.googlemail.com"

    con = imaplib.IMAP4_SSL(imap_url)


    login = con.login(user,password)

    result,section_data = con.select('INBOX')
    number_of_mails = section_data[0].decode('utf-8')


    result,data = con.fetch(section_data[0],'(RFC822)') #section_data[0] = no. of mails so we would get the last mail.


    raw = email.message_from_bytes(data[0][1])
    message = raw.get_payload(0).get_payload(None,True)

    message = message.decode("utf-8")
    
    return message

message = get_last_message()
message = clean_mail(message)
#print(message)


#########################  TRAINING OUR DATA (USING NAIVE BAYES CLASSIFIER)      ##################################

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
#print(mnb)

mnb.fit(x_vec,Y)
message = [message]
message_vec = cv.transform(message)



##############################       TESTING THE AQUIRED MESSAGE AND PRINTING THE OUTPUT    ##################################33
prediction = mnb.predict(message_vec)
if prediction[0] == 'ham':
	print(message)
else:
	ch = input("Message is spam. Do you want to see it? ->")
	if ch == 'y':
		print(message)
	else:
		pass
