import numpy as np
from tqdm import tqdm as Pb #progress bar

class model_NN:

    def __init__(self,layers): #initialization
        self.layer = layers
        self.layer_count = layers.shape[0]
        self.store ={}
        self.hyper_parameters= {} # dictionary parameters weights and biases
        self.costs = []

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def tanh(self,z):
        return np.tanh(z)

    def relu(self,z):
        A= 0
        if z>0 :
            A=z
        return A

    def softmax(self,z):
        shift = z-np.max(z)
        exp = np.exp(shift)
        return  exp / np.sum(exp,axis=1,keepdims=True)

    def der_sigmoid(self,z):
        return  (self.sigmoid(z) * (1-self.sigmoid(z)))

    def der_tanh(self,z):
        return (1-self.tanh(z)*self.tanh(z))

    def der_relu(self,z):
        val= 0
        if z>0:
            val=1
        return val

    def initialize_paramerters(self):
        previous_layer = self.store["A0"].shape[1] # A0(m,n) m is the number of images and n is the number of pixels per image
        for layer in range(self.layer_count): # 0,1
            self.hyper_parameters["W"+str(layer+1)]= np.random.randn(self.layer[layer,1],previous_layer)*0.0001 # W(units l ,units l-1)
            self.hyper_parameters["b"+str(layer+1)]= np.random.randn(self.layer[layer,1])*0.0001 # b(units l,1)
            previous_layer = self.layer[layer,1]

    def switch(self,arg):
        return {
            0:self.sigmoid,
            1:self.tanh,
            2:self.relu,
            3:self.softmax,
        }.get(arg,self.sigmoid)
    def der_switch(self,arg):
        return {
            0:self.der_sigmoid,
            1:self.der_tanh,
            2:self.der_relu,
            }.get(arg,self.der_sigmoid)

    def computeCost(self,y_hat,y): # cost = -sum(Y-logY_hat)/m
        return -np.mean(y*np.log(y_hat))

    def forword_prop(self):
        for layer in range(1,self.layer_count+1): # 1,2
            z= self.store["A"+str(layer-1)].dot(self.hyper_parameters["W"+str(layer)].T) + self.hyper_parameters["b"+str(layer)]
            self.store["A"+str(layer)]= self.switch(self.layer[layer-1,2])(z)

            #Z (layer l) = A (layer l-1) . W (layer l) + b (layer l)
            #A (layer l) = Activation_function(Z)

    def backward_prop(self):
        for bd_layer in reversed(range(1,self.layer_count +1)): # 2,1
            self.store["dW"+str(bd_layer)] = self.store["dz"+str(bd_layer)].T.dot(self.store["A"+str(bd_layer-1)])
            self.store["db"+str(bd_layer)] = np.sum(self.store["dz"+str(bd_layer)],axis=0)
        # dW (layer l) = (dz (layer l).T * A (layer l-1))
        # db (layer l) = sum (dz (layer l))

        if bd_layer>1: # no need for the first layer derivative operation
            self.store["dA"+str(bd_layer-1)]= self.store["dz"+str(bd_layer)].dot(self.hyper_parameters["w"+str(bd_layer)])
            self.store["dz"+str(bd_layer-1)]= self.der_switch(self.layer[bd_layer-1,2])(self.store["dA"+str(bd_layer-1)])
        # dA (layer l-1) = dz (layer l). w (layer l)
        # dz (layer l-1) = dericvative activation function (layer l-1) * dA (layer l-1)

    def Model_N_Network(self,X,Y,batch_size,epoch=50,learning_rate=0.01):
        batches_per_epoch = int(X.shape[0]/batch_size) # number of batches we have
        remaining_from_batch = X.shape[0]%batch_size
        if remaining_from_batch>0:
            print("remain:",remaining_from_batch)
        self.store["A"+str(0)] =X[0:batch_size,:] # A0 = X[0,:]
        self.initialize_paramerters()
        costs=[]
        outer= Pb(total=epoch,desc='Epoch',position=0,leave=None)
        for spin in range(epoch): # 1 epoch( one iteration ) covers all data
            inner= Pb(total=batches_per_epoch,desc='Batch',position=1,leave=None)
            for batch in range(batches_per_epoch):
                batch_start = batch*batch_size
                batch_end = batch_start+batch_size
                self.store["A"+str(0)] = X[batch_start:batch_end,:]
                self.forword_prop()
                self.store["dz"+str(self.layer_count)]= self.store["A"+str(self.layer_count)]-Y[batch_start:batch_end]
                # dz (last layer) = A (last layer) - Y only for last layer L
                self.backward_prop()
                for i in reversed(range(1,self.layer_count +1)):
                    self.hyper_parameters["W"+str(i)]=self.hyper_parameters["W"+str(i)]-(learning_rate/batch_size)*self.store["dw"+str(i)]
                    self.hyper_parameters["b"+str(i)]=self.hyper_parameters["b"+str(i)]-(learning_rate/batch_size)*self.store["db"+str(i)]
                # W (layer l) = W (layer l) - (learning_rate/batch_size) *dW (layer l)
                # b (layer l) = b (layer l) - (learning_rate/batch_size) *db (layer l)

                if batch % 100 == 0:
                    cost = self.computeCost(self.store["A"+str(self.layer_count)],Y[batch_start:batch_end,:])
                    costs.append(cost)
                inner.update(1)
            inner.close()
            outer.update(1)
        outer.close()
