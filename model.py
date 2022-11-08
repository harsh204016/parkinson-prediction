import os 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report , plot_confusion_matrix
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
seed = 42

file = open("metrics.txt","w")

def call_metrics(model=None,xtest=None,ytest=None):
    file.write(str(model)+"\n")
    file.write("Accuracy Score "+str(accuracy_score(ytest,model.predict(xtest)))+"\n")
    file.write(classification_report(ytest,model.predict(xtest))+"\n")
    plot_confusion_matrix(model, xtest, model.predict(xtest))  
    plt.savefig(str(model)+"confusion.png")

os_path = os.getcwd()
print("OS PATH",os_path)
data = pd.read_csv("parkinsons.data")

### seperating the X and y values ####
data.drop(["name"],axis=1,inplace=True)
X = data.drop(["status"],axis=1)
y = data["status"]

### Dividing the train and test data ###
xtrain,xtest,ytrain,ytest = train_test_split(X,y,shuffle=True,stratify=y,random_state=seed)

####################
## Model building ##
####################
logistic_model = LogisticRegression()
logistic_model.fit(xtrain,ytrain)


call_metrics(logistic_model,xtest,ytest)

####################
## Model Building ##
####################
randomforest_model = RandomForestClassifier(max_depth=5, random_state=seed)
randomforest_model.fit(xtrain,ytrain)

call_metrics(randomforest_model,xtest,ytest)


# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

xtrain,xtest,ytrain,ytest = train_test_split(X,y,shuffle=True,stratify=y,random_state=seed)
####################
## Model building ##
####################
logistic_model = LogisticRegression()
logistic_model.fit(xtrain,ytrain)

# file.write("After Upsampling\n")
# call_metrics(logistic_model,xtest,ytest)







