import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np


###Loading Data###

#import data
data = pd.read_csv('heart_failure.csv')

#view data columbs and types
print(data.info())

#view distribution of the target column 'death event'
print('Classes and number of values in the dataset', Counter(data['death_event']))

#create variable for label and features column
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

#convert the categorical feature values
x = pd.get_dummies(x) 


###Data Preprocessing###


#split the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


#scale the numeric training and test feature values
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


###Prepare Labels for classification###

#create label encoder instance
le = LabelEncoder()

# fit the encoder instance le to the training labels Y_train, while at the same time converting the training labels according to the trained encoder.
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.fit_transform(Y_test.astype(str))

#encode training and test labels labels into a binary vector
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)



###Create Model###


#create model instance with input, hidden and output layers 
model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],))) #input
model.add(Dense(12, activation = 'relu')) #hidden
model.add(Dense(2, activation = 'softmax')) #output
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compile model\
model.fit(X_train, Y_train, epochs = 100, batch_size = 16)#fit model
loss, acc = model.evaluate(X_test, Y_test)

###Generate  a classification report###

y_estimate = model.predict(X_test) #model prediction
y_estimate = np.argmax(y_estimate, axis=1) #
y_true = np.argmax(Y_test, axis=1) ##select the indices of the true classes vector
print(classification_report(y_true, y_estimate))

