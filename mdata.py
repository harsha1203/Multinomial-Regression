### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mdata = pd.read_csv("D:\Modules\Module 10 - Multinominal regression/mdata.csv")
mdata.head(10)

mdata1 = mdata.iloc[:,5:10]

mdata1.describe()
mdata1.prog.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x="prog",y="read",data=mdata1)
sns.boxplot(x="prog",y="write",data=mdata1)
sns.boxplot(x="prog",y="math",data=mdata1)
sns.boxplot(x="prog",y="science",data=mdata1)



# Scatter plot for each categorical choice of car
sns.stripplot(x="prog",y="read",jitter=True,data=mdata1)
sns.stripplot(x="prog",y="write",jitter=True,data=mdata1)
sns.stripplot(x="prog",y="math",jitter=True,data=mdata1)
sns.stripplot(x="prog",y="science",jitter=True,data=mdata1)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mdata1,hue="prog") # With showing the category of each car choice in the scatter plot
sns.pairplot(mdata1) # Normal

# Correlation values between each independent features
mdata1.corr()


train,test = train_test_split(mdata1,test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])

train_predict = model.predict(train.iloc[:,1:]) # Train predictions 
train_predict
test_predict = model.predict(test.iloc[:,1:]) # Test predictions

# Train accuracy 
accuracy_score(train.iloc[:,0],train_predict) # 60.62
# Test accuracy 
accuracy_score(test.iloc[:,0],test_predict) # 40.00



