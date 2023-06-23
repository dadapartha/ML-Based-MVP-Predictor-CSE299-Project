from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# importing dataset and preprocessing for model building
df = pd.read_csv('CSE299_DB.csv')
input = df.drop(columns=['MVP', 'Team Name', 'Player ID','Player Name'])
output = df['MVP']
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.2)

# training and creating DecisionTree model 
DT_model = DecisionTreeClassifier()
DT_model.fit(input_train, output_train)
pickle.dump(DT_model, open("DT_model.pkl", "wb"))

# training and creating KNN model 
KNN_model = KNeighborsClassifier()
KNN_model.fit(input_train, output_train)
pickle.dump(KNN_model, open("KNN_model.pkl", "wb"))

# training and creating LogisticRegression model 
LR_model = LogisticRegression()
LR_model.fit(input_train, output_train)
pickle.dump(LR_model, open("LR_model.pkl", "wb"))

# training and RandomForest tree model 
RF_model = RandomForestClassifier()
RF_model.fit(input_train, output_train)
pickle.dump(RF_model, open("RF_model.pkl", "wb"))

# training and AdaBoost tree model 
AB_model = AdaBoostClassifier()
AB_model.fit(input_train, output_train)
pickle.dump(AB_model, open("AB_model.pkl", "wb"))