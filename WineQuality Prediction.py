import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree

#importing dataset
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#head information of the wine dataset
print (data.head())

#seperating The data into features and labels. y is the label and X is the features
y = data.quality
X = data.drop('quality', axis=1)

#splitting into test and train data. Using 30% of the data to verify the predicted values by the model and 70% to train the model.

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

print("X_train:")
print(X_train.head())

#Train Data Preprocessing/Data Normalisation
X_train_scaled = preprocessing.scale(X_train)
print (X_train_scaled)

#Training the Algorithim/Training The Classfier
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

#Predicting the quality of the wine (labels)
confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence*100)

#obtaining labels
y_pred = clf.predict(X_test)


#Comparing The Predicted And Expected Labels
#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print (x[i])
    
#printing first five expectations
print("\nThe expectation:\n")
print (y_test.head())

#SpotChecking Algorithim- Using multiple machine learning algorithim to determine quality: Using Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt 

 # Finding correlations between each attribute of dataset with variable quality
correlations=data.corr()['quality'].drop('quality')
print(correlations) 

f,ax=plt.subplots(figsize=(10,6))

sns.heatmap(data.corr(),annot=True, ax=ax, cmap="coolwarm",fmt='.2f',linewidths=0.05)
f.subplots_adjust(top=0.93)
plt.show()


#outputing features whose correlation is above a certain threshold
def get_features(correlation_threshold):
	abs_corrs=correlations.abs()
	high_correlations=abs_corrs[abs_corrs> correlation_threshold].index.values.tolist()
	return (high_correlations)

#taking feature with correlation more than 0.05 as input x and quality as target variable y
features=get_features(0.05)
print(features)


input_x=data[features]
input_y=data['quality']

#creating training and testing set. 25% data used for training 75%
input_x_train,input_x_test,input_y_train,input_y_test=train_test_split(input_x,input_y,random_state=3)

# fitting linear regression to training data
regressor=LinearRegression()
regressor.fit(input_x_train,input_y_train)

print(regressor.coef_)

#predict the quality of the wine
train_pred=regressor.predict(input_x_train)
print(train_pred)
test_pred=regressor.predict(input_x_test)
print(test_pred)

#calculating root mean squared error

train_rmse=mean_squared_error(train_pred,input_y_train)**0.5
print(train_rmse)
test_rmse=mean_squared_error(test_pred,input_y_test)**0.5
print(test_rmse)

#rounding off the predicted values for test set
predicted_data=np.round_(test_pred)
print(predicted_data)

print("Mean Absolute Error:", metrics.mean_absolute_error(input_y_test, test_pred))
print("Mean Squared Error:", np.sqrt(metrics.mean_squared_error(input_y_test,test_pred)))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(input_y_test,test_pred)))

#displaying coefficients of each feature
coefficients=pd.DataFrame(regressor.coef_,features)
coefficients.columns=['coefficients']
print(coefficients)

