import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os

os.chdir("C:/Users/user/Desktop/credit card fraud/")

df = pd.read_csv("creditcard.csv")

print(df.head(20))

#dropping time
df = df.drop("Time", axis = 1)

# dividing our data into two parts, clean and dirty
fraud = df[df["Class"] ==1]
train_df = fraud.sample(frac = 0.8)
#adding clean transactions 80 percent to train data
clean = df[df["Class"] ==0]
train_df = pd.concat([train_df, clean.sample(frac = 0.8)], axis = 0)
#test dataframe contains all transactions not in train dataframe
a = df.index.tolist()
b = train_df.index.tolist()
b_set = set(b)

diff = [x for x in a if x not in b_set]
test_df = df[df.index.isin(diff)]

#shuffle datasets so that training will be random 
train_df = shuffle(train_df)
test_df = shuffle(test_df)


#Add our target features to y_train and y_test.
y_train = train_df.Class
y_test = test_df.Class


X_train = train_df.drop("Class", axis = 1)
X_test = test_df.drop("Class", axis = 1)

print(len(y_test))
print(len(X_test))
print(len(y_train))
print(len(X_train))

#scaling our data in order to feed it to neural model
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)


# Defining our classifier builder object to do all layers in once using layer codes from previous part

def classifier_builder ():
    classifier = Sequential()
    classifier.add(Dense(units = 10, activation='relu',kernel_initializer='uniform', input_dim = 29))
    #classifier.add(Dropout(p= 0.1))
    classifier.add(Dense(units = 10, activation='relu',kernel_initializer='uniform'))
    #classifier.add(Dropout(p= 0.1))
    classifier.add(Dense(units = 10, activation='relu',kernel_initializer='uniform'))
    #classifier.add(Dropout(p= 0.1))
    classifier.add(Dense(units = 10, activation='relu',kernel_initializer='uniform'))
    #classifier.add(Dropout(p= 0.1))
    classifier.add(Dense(units = 1, activation='sigmoid',kernel_initializer='uniform'))
    #classifier.add(Dropout(p= 0.1))
    classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return classifier



#Now we should create classifier object using our internal classifier object in the function above
classifier = KerasClassifier(build_fn= classifier_builder,
                             batch_size = 10,
                             nb_epoch = 100)
accuracies = cross_val_score(iestimator=classifier, 
                             X = X_train,
                             y = y_train,
                             cv= 10)

# finding the mean accuracy
mean = accuracies.mean()
variance = accuracies.std()



