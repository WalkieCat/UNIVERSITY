import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from keras.datasets import mnist # Importing MNIST datasets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 60k train and 10k test examples in a 28x28 pixel grid
print("Training set shape: ", x_train.shape)
print("Test set shape: ", x_test.shape)

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

x_train_flat = x_train_flat.astype('float32') / 255.0
x_test_flat = x_test_flat.astype('float32') / 255.0

model = LogisticRegression (
    solver='saga',
    multi_class='multinomial',
    max_iter=100,              
    #verbose=1,      #Show progress
    n_jobs=-1       
)

model.fit(x_train_flat, y_train)

# Evaluate
accuracy = model.score(x_test_flat, y_test)
print(f"Test accuracy: {accuracy: .4f}")

# Perform predictions on new data
y_pred = model.predict(x_test_flat)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)

print("\nClassification report: \n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.show()

correctIndices = np.where(y_pred == y_test)[0]
incorrectIndices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(10, 5))
for i, idx in enumerate(correctIndices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
    plt.axis('off')
plt.suptitle('Correct Predictions')
plt.show()

plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrectIndices[:10]):
    plt.subplot(2, 5, i  + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}", color='red')
    plt.axis('off')
plt.suptitle('Incorrect Predictions')
plt.show()
