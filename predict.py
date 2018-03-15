from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.naive_bayes import GaussianNB

from mlxtend.preprocessing import one_hot

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from sklearn.metrics import log_loss, precision_score, accuracy_score

import pandas as pd

import numpy as np

import sys



data = pd.read_csv("SFPD_Incidents_-_Current_Year__2017_.csv")



fulldata = data.copy()

del data['IncidntNum']

del data['Address']

del data['Location']

del data['PdId']

del data['Descript']



data['Category'].unique()



for i in range(len(data['Category'])):

	if data['Category'][i] in ["ROBBERY", "BRIBERY", "EXTORTION","STOLEN PROPERTY"]:

		data.set_value(i, 'Category', 0)

	elif data['Category'][i] in ["BURGLARY", "LARCENY/THEFT", "VEHICLE THEFT"]:

		data.set_value(i, 'Category', 1)

	elif data['Category'][i] in ["MISSING PERSON","RUNAWAY","KIDNAPPING"]:

		data.set_value(i, 'Category', 2)

	elif data['Category'][i] in ["DRUG/NARCOTIC" , "LIQUOR LAWS", "DRUNKENNESS", "DRIVING UNDER THE INFLUENCE"]:

		data.set_value(i, 'Category', 3)

	elif data['Category'][i] in ["ARSON"]:

		data.set_value(i, 'Category', 4)

	elif data['Category'][i] in ["EMBEZZLEMENT" ,"FRAUD", "FORGERY/COUNTERFEITING","BAD CHECKS"]:

		data.set_value(i, 'Category', 5)

	elif data['Category'][i] in ["SEX OFFENSES, FORCIBLE", "PROSTITUTION", "PORNOGRAPHY/OBSCENE MAT", "SEX OFFENSES, NON FORCIBLE"]:

		data.set_value(i, 'Category', 6)

	elif data['Category'][i] in ["ASSAULT"]:

		data.set_value(i, 'Category', 7)

	elif data['Category'][i] in ["SUICIDE"]:

		data.set_value(i, 'Category', 8)

	elif data['Category'][i] in ["VANDALISM", "LOITERING","DISORDERLY CONDUCT"]:

		data.set_value(i, 'Category', 9)

	elif data['Category'][i] in ["SECONDARY CODES","FAMILY OFFENSES"]:

		data.set_value(i, 'Category', 10)

	elif data['Category'][i] in ["WEAPON LAWS"]:

		data.set_value(i, 'Category', 11)

	elif data['Category'][i] in ["SUSPICIOUS OCC", "TRESPASS", "TREA"]:

		data.set_value(i, 'Category', 12)

	elif data['Category'][i] in ["NON-CRIMINAL", "OTHER OFFENSES", "GAMBLING"]:

		data.set_value(i, 'Category', 13)



for i in range(len(data["Resolution"])):

	if data["Resolution"][i] in ["ARREST", "BOOKED", "JUVENILE BOOKED", "UNFOUNDED","EXCEPTIONAL CLEARANCE", "NOT PROSECUTED","CLEARED-CONTACT JUVENILE FOR MORE INFO", "PSYCHOPATHIC CASE","ARREST", "CITED", "DISTRICT ATTORNEY REFUSES TO PROSECUTE", "LOCATED", "JUVENILE CITED"]:

		data.set_value(i, "Resolution", 1)

	else:

		data.set_value(i, "Resolution", 0)



#DELETE THOSE ROWS FROM MAIN DATABASE WHERE X, Y VALUES ARE IN DEGREES (90deg, 122deg)



data = data[data.Category != 'WARRANTS']

data = data[data.Category != 'RECOVERED VEHICLE']

data = data.reset_index(drop=True)



#X,Y: Location - to find range of values

#deleting location column as redundant with X, Y

data['Y'].max() #37.819975492297004

data['Y'].min() #37.707921903458598

data['X'].min() #-122.513642064265

data['X'].max() #-122.365565425353



date_data = data['Date']

date_data = date_data.str.split("/")



#Scope of Improvement: Address, Time, Date, descript



new_month = pd.get_dummies(pd.DatetimeIndex(data['Date']).month)

new_month.columns = ['month_' + str(col)  for col in new_month.columns]

data = data.join(new_month)



PdDistrict = data['PdDistrict']

new_pdDistrict = pd.get_dummies(PdDistrict)

data = data.drop('PdDistrict', axis=1)

data = data.join(new_pdDistrict)



new_days = pd.get_dummies(data['DayOfWeek'])

data = data.drop('DayOfWeek', axis=1)

data = data.join(new_days)



data = data.drop('Date', axis=1)



data['X'] = (data['X'] + 122) * 10000

data['Y'] = (data['Y'] - 37) * 10000



#Categorical variables: Resolution, PdDistrict, DoW



data_time = data["Time"]

for i in range(len(data_time)):

	data_time.set_value(i, data_time[i].replace(":",""))



data = data.drop('Time', axis=1)



train = data.sample(frac=0.6, random_state=100) #21686

test = data.drop(train.index) #14458



X_train = train.ix[:, train.columns != 'Category']

Y_train = list(train.Category.values)



X_test = test.ix[:, test.columns != 'Category']

Y_test = list(test.Category.values)





rfc = RandomForestClassifier()

rfc_model = rfc.fit(X_train, Y_train)

y_pred = rfc_model.predict(X_test)



precScore = precision_score(Y_test, y_pred, average="macro")

print "precision score", precScore



predProb = rfc_model.predict_proba(X_test)

print "y predicted", set(y_pred)

log_loss = log_loss(Y_test, predProb)

print "log loss", log_loss



acc = accuracy_score(Y_test, y_pred)

print "Accuracy is : ", acc





gnb = GaussianNB()

gnb_model = gnb.fit(X_train,Y_train)



y_p = gnb_model.predict(X_test)

pProb = gnb_model.predict_proba(X_test)

print "y predicted", set(y_p)

log_loss = log_loss(Y_test, pProb)



acc_score = accuracy_score(Y_test, y_p)

print "log loss", log_loss

print "acc score", acc_score







#######################################################

#S3VM --> 20% labeled , 80% unlabeled for train



train = data.sample(frac=0.6, random_state=100) #21686

l = train.sample(frac = 0.2, random_state=100)



X_train_l = l.ix[:, l.columns != 'Category']

Y_train_l = list(l.Category.values)



ul = train.drop(X_train_l.index)

X_train_ul = ul.ix[:, ul.columns != 'Category']



test = data.drop(train.index) #14458

X_test = test.ix[:, test.columns != 'Category']

Y_test = list(test.Category.values)



label_prop_model = LabelPropagation(kernel="knn")

labels = np.copy(Y_train_l)

label_prop_model.fit(X_train_l, labels)

label_prop_model.predict(X_train_ul)

#y = test.Category.values.reshape(-1,1)

label_prop_model.score(X_test, test.Category.values)



label_prop_model = LabelSpreading()

labels = np.copy(Y_train_l)

label_prop_model.fit(X_train_l, labels)

label_prop_model.predict(X_train_ul)

#y = test.Category.values.reshape(-1,1)

label_prop_model.score(X_test, test.Category.values)
