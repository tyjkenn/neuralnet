import neuralnet as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# prepares the iris dataset. Gives the data, targets and number of possible targets
def prep_iris():
    data = pd.read_csv("iris.data.txt", dtype=None,
                       names=["sepalLength", "sepalWidth", "petalLength", "petalWidth", "class"])
    data["class"] = data["class"].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
    target = data['class']
    data = data.drop("class", axis=1)
    return data, target, 3


# prepares the diabetes dataset. Gives the data, targets and number of possible targets
def prep_diabetes():
    data = pd.read_csv("diabetes.data.txt", dtype=None,
                       names=["timesPreg", "plasGlu", "bloodPress", "skinFold", "serumIns",
                             "bmi", "pedigree" , "age", "class"])
    target = data["class"]
    data = data.drop("class", axis=1)
    return data, target, 2

# Outputs all the weights of all the neurons
def debug_weights(net):
    for layerI in range(len(net.layers)):
        print("Layer {}".format(layerI + 1))
        for neuronI in range(len(net.layers[layerI])):
            print("\tNeuron {}".format(neuronI + 1))
            for weightI in range(len(net.layers[layerI][neuronI].weights)):
                print("\t\t{}. {}".format(weightI, net.layers[layerI][neuronI].weights[weightI]))


# Get choice from user
print("1. Iris")
print("2. Diabetes")
choice = input("Pick one:")

# Build neural net
if choice == '1':
    data, target, class_count = prep_iris()
    net = nn.NeuralNet(.1, 4, [4, 3])
else:
    data, target, class_count = prep_diabetes()
    net = nn.NeuralNet(.1, 8, [8, 2])

# normalize data and convert to python lists
data = (data - data.mean()) / data.std()
data = data.as_matrix().tolist()
target = list(target)

# split into training and testing
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.3, train_size=.7, shuffle=True)

# train each epoch
accuracies = []
epochs = 0
for epoch in range(300):
    epochs += 1
    for i in range(len(data_train)):
        net.train(data_train[i], target_train[i])
    score = 0.0
    totals = [0] * class_count
    for j in range(len(data_test)):
        result = net.predict(data_test[j])
        totals[result] += 1
        if result == target_test[j]:
            score += 1
    score /= len(data_test)
    count_str = ""
    for i in range(class_count):
        count_str += "Class {}: {}, ".format(i, totals[i])
    sys.stdout.write("\rEpoch: {}, Accuracy: {}%, {}".format(epoch, score * 100, count_str))
    accuracies.append(score * 100)
    if score > .9:
        break;

# visualize
plt.plot(range(epochs), accuracies)
plt.show()

# compare to existing implementation
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=4, random_state=1)

clf.fit(data_train, target_train)
predictions = clf.predict(data_test)
score = 0.0
for i in range(len(predictions)):
    if predictions[i] == target_test[i]:
        score += 1
score /= len(predictions)
print('')
print('Existing implementation score:{}%'.format(score*100))
