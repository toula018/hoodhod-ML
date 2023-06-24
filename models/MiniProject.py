from HoodHod import HoodHod
import numpy as np



data = {
    "feature_1": [True, False, True, True, False, False,True],
    "feature_2": [False, True, True, False, True, True, False],
    "feature_3": [True, False, True, True, True,  False, False],
    "feature_4": [False, True, False, True, True, False, True],
    "target": ["Yes", "Yes", "Yes", "Yes", "No", "No", "No"]
}
X = np.array(list(data.values())[:-1]).T.astype(int)
y = np.where(np.array(data["target"]) == "No", 0, 1)
features_names = list(data.keys())[:-1]
h = HoodHod.BinaryDecisionTree(X,y, features_names,criterion='entropy',random_state=42)
tree=h.fit()
X_test=[[1,1,0,0]]
y_pre=[[0]]
y_pred=h.predict(X_test)
print(f"The predicted value of binary decision tree model is : {y_pred}")
h.visualize_tree(tree)







data = {
    "feature_1": [True, False, True, False],
    "feature_2": [True, True, False, False],
    "target": ["No", "Yes", "No", "Yes"]
}
X = np.array(list(data.values())[:-1]).T.astype(int)
y = np.where(np.array(data["target"]) == "No", 0, 1)
features_names = list(data.keys())[:-1]

hoodhod =HoodHod.BinaryDecisionTree(X,y, features_names,criterion='entropy',random_state=42)
tree = hoodhod.fit()

X_test=[[1,1],
        [0,0],
        [1,0]]

y_pred=hoodhod.predict(X_test)
print(f"The predicted value of binary decision tree model is : {y_pred}")
hoodhod.visualize_tree(tree)


X = [1]
y = [2]

lr =HoodHod.SimpleLinearRegression()
lr.fit(X,y)

X_test=[3]
y_pred=lr.predict(X_test)

print(f"The predicted value of simple linear regression model is : {y_pred}")
X = [0,0,0,0]
y = [3,3,4,5]

lr =HoodHod.SimpleLinearRegression()
lr.fit(X,y)

X_test=[1]
y_pred=lr.predict(X_test)

print(f"The predicted value of simple linear regression model is : {y_pred}")







