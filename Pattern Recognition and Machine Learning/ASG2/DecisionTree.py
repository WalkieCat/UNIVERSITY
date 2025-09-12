import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import tree

breastCancerDataset = load_breast_cancer()
breastCancerDataset.keys()

print("terget names:", breastCancerDataset.target_names)
print("data shape:", breastCancerDataset.data.shape)
print("feature names:", breastCancerDataset.feature_names)
print("number of samples:", breastCancerDataset.target.shape[0])
breastCancerDataset.target

Xtrain, Xtest, ytrain, ytest = train_test_split(breastCancerDataset.data, breastCancerDataset.target, test_size=0.2, random_state=42)
print(Xtrain.shape)
print(Xtest.shape)

model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("Accuracy: ", metrics.accuracy_score(ytest, ypred))
print("Precision: ", metrics.precision_score(ytest, ypred, average='weighted'))
print("Recall: ", metrics.recall_score(ytest, ypred, average='weighted'))
print("F1 score: ", metrics.f1_score(ytest, ypred, average='weighted'))

report = classification_report(ytest, ypred, target_names=breastCancerDataset.target_names)
print(report)

params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,10)}
grs = GridSearchCV(model, param_grid=params, cv=5)
grs.fit(Xtrain, ytrain)
print("Best hyper parameters: ", grs.best_params_)
model = grs.best_estimator_
ypred = model.predict(Xtest)

print("Accuracy: ", metrics.accuracy_score(ytest, ypred))
print("Precision: ", metrics.precision_score(ytest, ypred, average='weighted'))
print("Recall: ", metrics.recall_score(ytest, ypred, average='weighted'))
print("F1 score: ", metrics.f1_score(ytest, ypred, average='weighted'))

fig = plt.figure(figsize=(15, 10))
tree.plot_tree(model, filled=True, feature_names=breastCancerDataset.feature_names, class_names=breastCancerDataset.target_names)
plt.show()