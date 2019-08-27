import numpy as np 
import imaplib,email

user = "gotravel.agsr@gmail.com"
password = "AbhishekSachin"
imap_url = "imap.googlemail.com"

# Removing security from our account and enabling the less secure account
# Create the connection
con = imaplib.IMAP4_SSL(imap_url)

# Each of the con functions will return 2 values. One is the result and other is the data.

########### Login into our account   ###########
login = con.login(user,password)
#print(login)

##########   Now after login we will parse through the email sections.  #########
result,section_data = con.select('INBOX')
number_of_mails = section_data[0].decode('utf-8')

######   Now parsing through the section contents  ############

#result,data = con.fetch(b'105','(RFC822)')
result,data = con.fetch(section_data[0],'(RFC822)') #section_data[0] = no. of mails so we would get the last mail.

#print(result)
#print(data)



#DATA OBTAINED IS IN BYTES
#Now fetching the message from the bytes.

raw = email.message_from_bytes(data[0][1])
message = raw.get_payload(0).get_payload(None,True)
#print(message)
#print(type(message))

#CONVERTING BYTES INTO STRING
message = message.decode("utf-8")
print(message)
print(type(message))


