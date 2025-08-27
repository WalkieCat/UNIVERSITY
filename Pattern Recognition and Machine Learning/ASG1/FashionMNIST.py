import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from keras.datasets import fashion_mnist

# Loading the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training set size: ", X_train.shape)
print("Testing set size: ", X_test.shape)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

#Flatten the data and typecast
X_train_flat =X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

# Define a 5 fold K-Fold validation
kF = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models with L2 regularization
modelL2 = LogisticRegression (
    penalty='l2',
    max_iter=500, 
    C=1.0, 
    random_state=42
)

# Fitting and predicitng
modelL2.fit(X_train_flat, y_train)
y_pred = modelL2.predict(X_test_flat)

# Evaluate model
# Metrics: Accuracy, confusion matrix, classification report, cross-validation score
accuracy = modelL2.score(X_test_flat, y_test)
print(f"Model accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix: \n", cm)

print("\nClassification report: \n", classification_report(y_test, y_pred))

cvScore = cross_val_score(modelL2, X_test_flat, y_test, scoring="accuracy", n_jobs=-1)
print(f"\nCross validations score on 5 fold: {cvScore.mean():.4f} +/- {cvScore.std():.4f}")

# Visualize the first 10 correct prediction
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    plt.axis('off')


# Compare where the model predicted correct and incorrect
correctIndices = np.where(y_pred == y_test)[0]
incorrectIndices = np.where(y_pred != y_test)[0]

# Plot correct and incorrect predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(correctIndices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
    plt.axis('off')
plt.suptitle('Correct Predictions')


plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrectIndices[:10]):
    plt.subplot(2, 5, i  + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}", color='red')
    plt.axis('off')
plt.suptitle('Incorrect Predictions')

# Plot all graph at once
plt.show()