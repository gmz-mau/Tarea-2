
import random
import numpy as np
from tensorflow.keras.datasets import mnist

#Tomamos un dataset de keras con datos predeterminados para datos de entrenamiento y prueba
dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

#Definimos una clase
class Network(object):

    """Aquí vamos a definir la red, dando un valor a sizes, el cual será una lista
     donde cada elemento de ella, sera una capa de neuronas, siendo la primera las
     neuronas de "entrada" las de enmedio las capas ocultas y la última la capa de salida """
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)     #Cuantas capas hay
        self.sizes = sizes
        #No se le agrega un bias a la primera capa ya que son los valores de entrada
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  #Se definen los bias aleatoriamente de una normal(0,1)
        self.weights = [np.random.randn(y, x)                     #Igual que los bias, solo que para los pesos
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        "Definimos la salida de una capa y la entrada de la siguiente"
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    
    """Definimos el Stochastic Gradient Descent"""
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        training_data = list(training_data) #Datos de entrenamiento
        n = len(training_data)

        if test_data:        #Si se agrega datos de prueba se realiza
            test_data = list(test_data)
            n_test = len(test_data)     

        for j in range(epochs):
            random.shuffle(training_data)     #Se ordena aleatoriamente la lista de los datos de entrenamiento
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]     #Se realiza entrenamiento con cada "pedazo" del dataset (mini batch)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)    #Se actualiza el "pedazo" a usar en el entrenamiento con un cierto learning rate
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
    
    
    
    """Se actualizan los bias y los pesos usando backpropagation para obtener
    una mejor precisión, "castigando" los pesos o bias, así para el siguiente
    mini batch"""
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    "Backpropagation para el castigo de los pesos y bias"
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # La activación para el feedforward
        activation = x
        activations = [x] # Cada activación de cada capa
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    "Aqui observamos los resultados de los datos de prueba para las épocas"
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

"Función sigmoide"
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#One-hot encoding, obtenemos una capa de 0's y 1
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#Convertimos una imagen de nxm dimensiones a una matriz de 784x1 normalizada
x_train_flattened = [np.reshape(x, (784, 1))/255. for x in x_train]
x_test_flattened = [np.reshape(x, (784, 1))/255. for x in x_test]

#Aplicamos el one-hot encoding
y_train_categorical = [vectorized_result(y) for y in y_train]

training_data = list(zip(x_train_flattened, y_train_categorical))
test_data = list(zip(x_test_flattened, y_test))

#Datos para entrenar la Red Neuronal
net = Network([784, 40, 10])
net.SGD(training_data, 40, 10, 2.0, test_data=test_data)


