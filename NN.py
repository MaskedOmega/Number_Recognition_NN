import numpy as np
import pandas as pd

class nuerel_network():
    def __init__(self,xs,ys,shape):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.shape = [self.xs.shape[1]] + shape + [self.ys.shape[1]]

        self.weights = []
        self.bias = []

        self.__init_params__()
    
    def __init_params__(self):
        for i in range(len(self.shape)-1):
            self.weights.append(np.random.randn(self.shape[i+1],self.shape[i]))
            self.bias.append(np.random.randn(1,self.shape[i+1]))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self,x):
        return np.maximum(0, x)

    def relu_derivative(self,x):
        return (x > 0).astype(float)    
    
    def foward_pass(self,x):
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.bias):
            z = np.dot(activations[-1], w.T) + b
            zs.append(z)
            activations.append(self.sigmoid(z))
            #print(activations[-1])
        return activations, zs
    
    def backward_pass(self, activations, y):
        deltas = [None] * len(self.weights)
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.bias)

        error = activations[-1] - y
        deltas[-1] = error * self.sigmoid_derivative(activations[-1])

        for i in range(len(deltas) -2, -1, -1):
            deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1]) * self.sigmoid_derivative(activations[i + 1])

        for i in range(len(self.weights)):
            grads_w[i] = np.dot(deltas[i].T, activations[i])
            grads_b[i] = np.sum(deltas[i], axis=0, keepdims=True)
        
        for i in range(len(self.weights)):
            self.weights[i] -= 0.01 * grads_w[i]
            self.bias[i] -= 0.01 * grads_b[i]

    def predict(self, x):
        activations, _ = self.foward_pass(np.array(x))
        return activations[-1]

    def train(self, epochs=10):
        for epoch in range(epochs):
            activations, zs = self.foward_pass(self.xs)
            self.backward_pass(activations, self.ys)
            if epoch % 1000 == 0:
                loss = np.mean((self.ys - activations[-1]) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")




data_quant = 100
csvFile = pd.read_csv('mnist_test.csv')
rows = csvFile.iloc[:data_quant]


x = [[0,0],[0,1],[1,0],[1,1]]
y = [[1],[0],[0],[1]]


x = rows.iloc[:, 1:785].values
y = rows.iloc[:, 0].values.reshape(-1, 1)

def y_group(y):
    size = int(np.max(y)) + 1
    arr = np.zeros((len(y), size))
    for i in range(len(y)):
        arr[i][y[i][0]] = 1
    return arr

y = y_group(y)

print("Preprocessing complete. x shape:", x.shape, "y shape:", y.shape)

number = nuerel_network(x,y,[50,10,5])
number.train(10000)

count = 0
for sample in x:
    print(f"Predicted: {np.argmax(number.predict([sample]))}, Actual: {y[count]}")
    count+=1
    

