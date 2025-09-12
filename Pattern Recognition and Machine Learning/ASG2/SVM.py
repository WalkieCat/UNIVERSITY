import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

faces = fetch_lfw_people(min_faces_per_person=80)

fig, ax = plt.subplots(3, 4, figsize=(8,8))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel = faces.target_names[faces.target[i]])
plt.tight_layout()
plt.show()
    
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, train_size=0.8, random_state=0)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

model = SVC()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("Accuracy: ", metrics.accuracy_score(ytest, ypred))
print("Precision: ", metrics.precision_score(ytest, ypred, average='weighted'))
print("Recall: ", metrics.recall_score(ytest, ypred, average='weighted'))
print("F1 score: ", metrics.f1_score(ytest, ypred, average='weighted'))
print("Classification report: ", classification_report(ytest, ypred, target_names = faces.target_names))

cfm = confusion_matrix(ytest, ypred)  
plt.figure(figsize=(7, 5))
sns.heatmap(cfm, square=True, annot=True, fmt='d', cbar=False, xticklabels = faces.target_names, yticklabels = faces.target_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Heat map')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=faces.target_names)
fig, ax = plt.subplots(figsize=(7,5))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()

params = model.get_params()
#print(params) 

model = SVC()
paramGrid = [
    {
        'C': [0.5, 0.1, 1, 5, 10],
        'kernel': ['linear'],
        'class_weight': ['balanced']},
    {
        'C': [0.5, 0.1, 1, 5, 10],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.05, 0.5],
        'kernel': ['rbf'],
        'class_weight': ['balanced']}]

grs = GridSearchCV(model, paramGrid)
grs.fit(Xtrain, ytrain)
print('Best Hyper parameter: ', grs.best_params_)
modelBest = grs.best_estimator_
ypred = modelBest.predict(Xtest)
print("Accuracy: ", metrics.accuracy_score(ytest, ypred))
print("Precision: ", metrics.precision_score(ytest, ypred, average='weighted'))
print("Recall: ", metrics.recall_score(ytest, ypred, average='weighted'))
print("F1 score: ", metrics.f1_score(ytest, ypred, average='weighted'))

plt.figure(figsize=(8,6))
sns.heatmap(cfm, square=True, annot=True, fmt='d', cbar=False, xticklabels = faces.target_names, yticklabels = faces.target_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Heat map')
plt.show()

fig, ax = plt.subplots(4, 6, figsize=(10,10))
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_xlabel("true:" +faces.target_names[ytest[i]].split()[-1] +
                   "\npred:" +faces.target_names[ypred[i]].split()[-1],
                   color='black' if ypred[i] == ytest[i] else 'red')
fig.suptitle("Predicted Names: Incorrect Labels in Red", size='14')
plt.show()