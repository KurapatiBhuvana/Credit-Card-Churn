

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msn
from collections import Counter





df  = pd.read_csv("credit_card_churn.csv")
#df.head()


# To get numerical values the catagorical columns need to be chnaged. Hence label encoding is done 
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

#Preprocessing to transform categorial to numerical to data pridiction
df_categorical = df[['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']]

#df_categorical.head()

#df_numerical = df[['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit','Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]
#df_numerical.head()

le = LabelEncoder()

# Encode the categorical variable
df['Gender'] = le.fit_transform(df['Gender'])

# Print the encoded categories




# Encode the categorical variable
df['Education_Level'] = le.fit_transform(df['Education_Level'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Income_Category'] = le.fit_transform(df['Income_Category'])
# Print the encoded categories


#print(df.head(25))

"""##Merge categorical and numerical dataframe"""

df_all = pd.concat([df['Gender'], df['Education_Level'],df['Marital_Status'],df['Income_Category'],df[['Customer_Age', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit','Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]], axis=1)
#df_all.head(22)

feature_names = df_all.columns
X = pd.DataFrame(df_all, columns=feature_names)
y = df['Attrition_Flag']
#X

le = LabelEncoder()
y = le.fit_transform(y)
#y[21]

"""##Test Train Split"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



target_names = ['Attrited Customer', 'Existing Customer']

parameters_randomforest = {'n_estimators':range(10,400,5), 'max_depth':range(2,8,2)}





"""##Gradient Boosting"""

clf_gb = GradientBoostingClassifier()



clf_gb.fit(X_train, y_train)
y_pred_gb = clf_gb.predict(X_test)


def ConGen(s):
  if(s == 'M'):
    return 1
  return 0

def ConEdu(s):
  if(s == 'High School'):
    return 3
  elif(s == 'Graduate'):
    return 2
  elif(s == 'Post-Graduate'):
    return 5
  elif(s == 'College'):
    return 0
  elif(s == 'Unknown'):
    return 6
  elif(s == 'Uneducated'):
    return 5
  return 1

def ConMar(s):
   if(s == 'Married'):
    return 1
   elif(s == "Divorced"):
    return 0
   elif(s == 'Unknown'):
    return 3
   return 2

def ConInc(s):
  if(s == '$60K-$80K'):
    return 2
  elif(s == 'Less than $40K'):
    return 4
  elif(s == '$80K-$120K'):
    return 3
  elif(s == '$40K-$60K'):
    return 1
  elif(s == '$120K+'):
    return 0
  return 5





def predict_creditCard_Churn(gen,edu,mar,inc,age):
    
    new_customer = []
    new_customer.append(ConGen(gen))
    new_customer.append(ConEdu(edu))
    new_customer.append(ConMar(mar))
    new_customer.append(ConInc(inc))
    new_customer.append(age)
    
    temp = [36	,	3.81,	2.341,2.45	,8631.95	,1162.81	,7469.13	,0.759	,4404.08,	64.85	,0.712,	0.27]
    new_customer.extend(temp)
    empty_df = pd.DataFrame(columns=df_all.columns)
    empty_df.loc[len(empty_df)] = new_customer
    prediction =clf_gb.predict(empty_df)
    if prediction == 1:
      return "Customer is likely to churn."
    else:
      return "Customer is not likely to churn."


import tkinter as tk
import PIL
from PIL import ImageTk,Image
from tkinter import ttk
from tkinter import messagebox
import tkinter as tk

from tkinter import *


from tkinter import Frame, Label, Entry, Button, GROOVE

import numpy as np




# Function to get predictions
def get_prediction():
    # Get the user input from the entry fields
    gender = e1.get()
    education = e2.get()
    marital_status = e3.get()
    income = e4.get()
    age = e5.get()

    output_data = predict_creditCard_Churn(gender, education, marital_status, income, age)
    show_output(output_data)
    output_display="OUTPUT"
    show_output()(output_display)

    
def show_output(output_data):
  # Create a new window
  output_window = Toplevel(root)
  output_window.title("Output")
  output_label = tk.Label(output_window, text="OUTPUT: \n\n\n" + output_data, font=("times new roman", 35))
  output_label.pack(pady=20)
  
  


# Function to clear the entry fields
def clear_fields():
    e1.delete(0, tk.END)
    e2.delete(0, tk.END)
    e3.delete(0, tk.END)
    e4.delete(0, tk.END)
    e5.delete(0, tk.END)


# Create the GUI
root = tk.Tk()
root.title("Credit Card Churn Prediction")
root.geometry("500x500")

# Load the background image
bg_image = PIL.Image.open("C:/Users/KBhuv/OneDrive/Desktop/Ml/credit.jpg")
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label widget for the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0,y=0,relwidth=1,relheight=1)
bg_label.lower()





# Set the size of the window
window_height = 400
window_width = 500
screen_height = root.winfo_screenheight()
screen_width = root.winfo_screenwidth()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


# Add a label to the top of the window
title_label = Label(root, text="Credit Card Churn Prediction", font=("Times New Roman", 45), fg = "white", bg = "black")
#title_label.place(relx=0.5, rely=0.1, anchor="w")
#your other label or button or ...

title_label.pack(padx = 0,pady=80)

# Create a frame for the input fields
input_frame = Frame(root, bg = "black")
input_frame.pack(padx=10, pady=10)

# Add labels and entry fields to the input frame
l1 = Label(input_frame, text="Customer Gender:", font=("Times New Roman", 20),bg = "black",fg = "darkorange")
l1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
e1 = Entry(input_frame, width=30)
e1.grid(row=0, column=1, padx=10, pady=10)

l2 = Label(input_frame, text="Customer Education:", font=("Times New Roman", 20),bg = "black",fg = "red")
l2.grid(row=1, column=0, padx=10, pady=10, sticky="w")
e2 = Entry(input_frame, width=30)
e2.grid(row=1, column=1, padx=10, pady=10)

l3 = Label(input_frame, text="Marital Status:", font=("Times New Roman", 20),bg = "black",fg = "darkorange")
l3.grid(row=2, column=0, padx=10, pady=10, sticky="w")
e3 = Entry(input_frame, width=30)
e3.grid(row=2, column=1, padx=10, pady=10)

l4 = Label(input_frame, text="Income:", font=("Times New Roman", 20),bg = "black",fg = "red")
l4.grid(row=3, column=0, padx=10, pady=10, sticky="w")
e4 = Entry(input_frame, width=30)
e4.grid(row=3, column=1, padx=10, pady=10)

l5 = Label(input_frame, text="Age:", font=("Times New Roman", 20),bg = "black",fg = "darkorange")
l5.grid(row=4, column=0, padx=10, pady=10, sticky="w")
e5 = Entry(input_frame, width=30)
e5.grid(row=4, column=1, padx=10, pady=10)

# Add a button to make predictions
button_frame = Frame(root)
button_frame.pack(pady=10)
button = Button(button_frame, text="Predict", command=get_prediction, font=("Times New ", 23), relief=GROOVE, bg = "black", fg = "white")
button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_fields, font=("Times New ", 23), relief=GROOVE, bg = "black", fg = "white")
clear_button.place(relx=0.47, rely=0.9, anchor="sw")

'''clear_button= Frame(root)
clear_button.pack(pady=10)
clrbutton = Button(button_frame, text="clear", command=clear_fields, font=("Georgia", 23), relief=GROOVE, bg = "white", fg = "black")
clrbutton.pack()'''

'''clear_button = tk.Button(root, text="Clear", command=clear_fields)
clear_button.place(anchor="center")
clear_button.pack(pady= 20)'''




'''clear_button = tk.Button(root, text="Clear", command=clear_fields)
clear_button.place(anchor="center")
clear_button.pack(pady= 20)'''
# Add a label to display the output

#output_label = Label(root, text="", font=("Helvetica", 14), bg="light blue")
#output_label.pack(pady=10)



root.mainloop()
