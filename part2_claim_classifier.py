import numpy as np
import pickle
import torch
import torch.nn as nn
import pandas as pd
import random
from sklearn import preprocessing, metrics


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.number_of_neurons = 3
        self.hidden1 = nn.Linear(9, 9)
        self.hidden2 = nn.Linear(9, 9)
        self.hidden3 = nn.Linear(9, 9)
        self.hidden4 = nn.Linear(9, 9)
        self.hidden5 = nn.Linear(9, 9)
        self.out = nn.Linear(9, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.hidden4(x)
        x = self.activation(x)
        x = self.hidden5(x)
        x = self.activation(x)
        x = self.out(x)
        return x


class ClaimClassifier:

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.net = None

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        X = X_raw.to_numpy()
        scaler = preprocessing.MinMaxScaler()
        return scaler.fit_transform(X)

    def fit(self, X_raw, y_raw, learning_rate, num_epochs):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        Y = y_raw.to_numpy()
        X_clean = self._preprocessor(X_raw)

        training_set = []
        for i, x in enumerate(X_clean):
            training_set.append((x, Y[i]))

        net = Net()
        net = net.float()

        batch_size = 100
        # learning_rate = learning_rate
        # num_epochs = num_epochs
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

        train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
        losses = []
        for i in range(num_epochs):
            for j, (x, labels) in enumerate(train_loader):
                optimiser.zero_grad()
                outputs = net.forward(x.float())
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimiser.step()
                losses.append(loss.data)
            print("Epoch: {}/{}, Loss: {}".format(i + 1, num_epochs, loss.data))

        self.net = net
        self.save_model()

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE
        model = load_model()
        X_clean = self._preprocessor(X_raw)

        # dummy label data for dataloader
        test_set = []
        for i, x in enumerate(X_clean):
            test_set.append((x, 0))

        test_loader = torch.utils.data.DataLoader(dataset=test_set)
        predictions = []
        for i, (x, y) in enumerate(test_loader):
            output = model.net.forward(x.float())
            # print(output.data)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.numpy()[0])
        return np.asarray(predictions)

    def evaluate_architecture(self, ground_truths, predictions):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        matrix = metrics.confusion_matrix(ground_truths, predictions)
        print("Confusion matrix:")
        print(matrix)
        accuracy = metrics.accuracy_score(ground_truths, predictions)
        print("Accuracy: {}".format(accuracy))
        (p, r, f1, _) = metrics.precision_recall_fscore_support(ground_truths, predictions)
        print("Precision (0): {}, Precision (1): {}".format(p[0], p[1]))
        print("Recall (0): {}, Recall (1): {}".format(r[0], r[1]))
        print("F1 Score (0): {}, F1 Score (1): {}".format(f1[0], f1[1]))
        return accuracy

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


def ClaimClassifierHyperParameterSearch(classifier, dataset):
    # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters


def main():
    original_data = pd.read_csv("part2_training_data.csv")
    main(original_data)


def main(original_data):
    # shuffle
    np.random.shuffle(original_data.values)

    # upsample to create balanced dataset
    up_sampled = [row for (index, row) in original_data.iterrows() if row["made_claim"] == 0]
    zero_count = len(up_sampled)

    ones = [row for (index, row) in original_data.iterrows() if row['made_claim'] == 1]

    for row in ones:
        up_sampled.append(row)

    one_count = len(ones)
    while one_count < zero_count:
        index = random.randint(0, len(ones) - 1)
        row = ones[index]
        up_sampled.append(row)
        one_count += 1
    data = pd.DataFrame(up_sampled)

    np.random.shuffle(data.values)

    training_set_boundary = int(0.6 * len(data))
    validation_set_boundary = int(0.8 * len(data))
    training_set = data[:training_set_boundary]
    validation_set = data[training_set_boundary:validation_set_boundary]
    test_set = data[validation_set_boundary:]

    X = training_set.drop(columns=["claim_amount", "made_claim"])
    y = training_set["made_claim"]

    X_validation = validation_set.drop(columns=["claim_amount", "made_claim"])
    y_validation = validation_set["made_claim"]

    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    num_epochs = [1, 5, 10, 50]

    best_lr = None
    best_epoch = None
    accuracy = 0

    for learning_rate in learning_rates:
        for epoch in num_epochs:
            print("Learning rate")
            print(learning_rate)
            print("Num epochs")
            print(epoch)
            classifier = ClaimClassifier()
            classifier.fit(X, y, learning_rate, epoch)
            predictions = classifier.predict(X_validation)
            curr_accuracy = classifier.evaluate_architecture(y_validation.to_numpy(), predictions)
            if curr_accuracy > accuracy:
                best_epoch = epoch
                best_lr = learning_rate
                accuracy = curr_accuracy

    print("========= Best results ==============")
    print(best_lr)
    print(best_epoch)
    print(accuracy)

    X_test = test_set.drop(columns=["claim_amount", "made_claim"])
    y_test = test_set["made_claim"]


if __name__ == "__main__":
    main()
