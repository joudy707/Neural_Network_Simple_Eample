# Iris Flower Classification with a Simple Neural Network

A complete introductory deep-learning project that uses PyTorch to classify Iris flowers into three species from four measured features. The project demonstrates the full machine-learning workflow: loading data, preprocessing, splitting the dataset, building a feed-forward neural network, training with backpropagation, evaluating on unseen data, predicting new samples, and saving/loading model parameters.

## Project Overview

The model predicts one of the following Iris species:

- **Setosa**: class `0`
- **Versicolor**: class `1`
- **Virginica**: class `2`

Each observation contains four numerical features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

This is a supervised, multiclass classification problem. During training, the network receives feature vectors together with their known class labels and learns a function that maps the measurements to class scores.

## Project Structure

```text
AI_Eng_proj/
|-- README.md
`-- NN/
    `-- NeuralNetwork.ipynb
```

The complete implementation and experiments are contained in [NN/NeuralNetwork.ipynb](NN/NeuralNetwork.ipynb).

## Dataset

The notebook loads the Iris dataset from a public CSV source:

```text
https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
```

The dataset contains 150 samples, 4 input features, and 3 target classes. The species names are converted to integer labels because PyTorch's `CrossEntropyLoss` expects class indices represented as integer values.

The label mapping is:

```python
setosa      -> 0
versicolor  -> 1
virginica   -> 2
```

## Methodology

The notebook follows these stages:

1. Import PyTorch and supporting libraries.
2. Define a neural-network architecture.
3. Load the Iris data into a pandas DataFrame.
4. Convert categorical species names into numerical labels.
5. Split the data into training and test sets.
6. Convert features and labels into PyTorch tensors.
7. Train the network using cross-entropy loss and the Adam optimizer.
8. Evaluate the trained model on the unseen test set.
9. Predict the class of new Iris measurements.
10. Save and reload the trained model parameters.

## Neural-Network Architecture

The model is a fully connected feed-forward neural network implemented by the `Model` class:

```text
Input layer:       4 features
Hidden layer 1:    8 neurons + ReLU
Hidden layer 2:    9 neurons + ReLU
Output layer:      3 class scores
```

The architecture is implemented using three linear transformations:

```python
self.fc1 = nn.Linear(in_features, h1)
self.fc2 = nn.Linear(h1, h2)
self.out = nn.Linear(h2, out_features)
```

For an input vector $x$, the forward pass is:

$$
h_1 = ReLU(W_1x + b_1)
$$

$$
h_2 = ReLU(W_2h_1 + b_2)
$$

$$
z = W_3h_2 + b_3
$$

The output vector $z$ contains three logits, one for each Iris species. A logit is an unnormalized class score. The largest logit determines the predicted class.

### ReLU Activation

The Rectified Linear Unit is defined as:

$$
ReLU(a) = max(0, a)
$$

ReLU introduces non-linearity, allowing the network to learn decision boundaries that cannot be represented by a single linear transformation.

## Data Preprocessing

The input features are converted to floating-point tensors:

```python
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
```

The target labels are converted to integer tensors:

```python
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
```

`LongTensor` is required because `nn.CrossEntropyLoss` interprets each target as an integer index for the correct class.

The data is divided using:

```python
train_test_split(x, y, test_size=0.2, random_state=41)
```

This produces approximately 80% training data and 20% test data. The fixed random state makes the split reproducible.

## Training Procedure

The model is trained for 100 epochs. One epoch is one complete pass through the training set.

At every epoch:

1. The training features are passed through the model.
2. The predicted logits are compared with the true labels.
3. The loss gradient is calculated by backpropagation.
4. The optimizer updates the model parameters.

The training loss is calculated with:

```python
criterion = nn.CrossEntropyLoss()
```

For a single sample, cross-entropy loss is:

$$
L = -log(p_y)
$$

where $p_y$ is the predicted probability assigned to the correct class. In PyTorch, `CrossEntropyLoss` combines the softmax operation and the negative log-likelihood calculation internally, so the network should output raw logits rather than manually applying softmax.

The optimizer is:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

Adam adapts the learning rate for each parameter using estimates of the first and second moments of the gradients. The learning rate controls the size of each parameter update.

## Evaluation

After training, the model is evaluated on `x_test`, which contains samples that were not used during optimization:

```python
with torch.no_grad():
    y_eval = model(x_test)
    loss = criterion(y_eval, y_test)
```

`torch.no_grad()` disables gradient tracking during inference. This reduces memory usage and prevents unnecessary gradient computation.

### Computing Correct Predictions

For each test sample, the model returns three logits. The predicted class is the index of the largest logit:

```python
prediction = y_val.argmax().item()
```

The number of correct predictions is counted by comparing the prediction for the current sample with its corresponding label:

```python
if prediction == actual:
    correct += 1
```

The accuracy is calculated as:

$$
Accuracy = \frac{Number\ of\ correct\ predictions}{Total\ number\ of\ test\ samples} \times 100
$$

It is important to use the output for the current sample inside the loop. Comparing the full batch output `y_eval` instead of the current output `y_val` can produce an incorrect count, often resulting in zero correct predictions.

## Predicting New Flowers

The notebook also tests the trained network with new measurements. For example:

```python
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
```

The model returns three logits:

```python
with torch.no_grad():
    scores = model(new_iris)
    predicted_class = scores.argmax().item()
```

The predicted class is the index with the highest score. For a user-facing application, this integer can be mapped back to the species name:

```python
class_names = ['Setosa', 'Versicolor', 'Virginica']
print(class_names[predicted_class])
```

The same procedure is used for the second example, `newer_iris`.

## Saving and Loading the Model

The trained parameters are saved with:

```python
torch.save(model.state_dict(), 'my_iris_model.pt')
```

A new model instance can load these parameters:

```python
new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))
new_model.eval()
```

`state_dict()` stores the learned weights and biases. The architecture must be recreated before loading the parameters because the saved state does not by itself define the complete model class.

Calling `eval()` switches the model to evaluation mode. This is essential for architectures containing layers such as dropout or batch normalization. The current network does not use those layers, but using evaluation mode is still the correct inference convention.

## Installation

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install torch pandas matplotlib scikit-learn jupyter
```

Depending on the operating system and hardware, PyTorch may have a platform-specific installation command. The official PyTorch installation selector can be used when CUDA or another accelerator is required.

## Running the Project

1. Open the project folder in VS Code.
2. Open `NN/NeuralNetwork.ipynb`.
3. Select a Python interpreter with the required packages installed.
4. Run the notebook cells from top to bottom.
5. Inspect the loss plot, test loss, accuracy, new-sample predictions, and model save/load results.

Running cells in order is important because later cells depend on variables and model parameters created earlier.

## Reproducibility

The notebook sets a PyTorch seed:

```python
torch.manual_seed(41)
```

It also uses `random_state=41` for the dataset split. These settings make the initialization and split repeatable in the same software environment. Exact results can still vary across operating systems, PyTorch versions, devices, and numerical backends.

## Scientific Interpretation

The network learns a nonlinear approximation:

$$
f: R^4 -> R^3
$$

The four-dimensional input represents the flower measurements, and the three-dimensional output represents class logits. During optimization, the model changes its weights to increase the score of the correct class while decreasing the relative scores of incorrect classes.

The Iris dataset is relatively small and well studied. Its classes are not equally difficult: Setosa is usually easier to separate from the other classes, while Versicolor and Virginica have more overlapping measurements. Therefore, a high overall accuracy does not necessarily mean that every class is predicted equally well. A more complete evaluation could include a confusion matrix, per-class precision, recall, and F1-score.

## Limitations and Possible Improvements

This project is intentionally compact and educational. For a stronger production-quality experiment, consider:

- Standardizing the input features.
- Using stratified splitting to preserve class proportions.
- Adding a validation set for hyperparameter selection.
- Training for more epochs with early stopping.
- Reporting a confusion matrix and per-class metrics.
- Comparing the neural network with logistic regression, a decision tree, or a support-vector machine.
- Saving the class mapping together with the model.
- Validating the input shape and feature order before inference.
- Avoiding variable reuse, such as using `x` both for feature data and for a display label.
- Loading model files with the appropriate PyTorch security and device options for the deployed environment.

## Technologies

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Learning Objectives

This project provides practical experience with:

- Defining custom PyTorch models.
- Understanding layers, neurons, weights, and biases.
- Applying activation functions.
- Using logits and cross-entropy loss for multiclass classification.
- Training with gradient descent and backpropagation.
- Evaluating predictions correctly.
- Performing inference on new data.
- Persisting and restoring learned model parameters.
