import numpy as np
import torch
from torch import nn



#feed-forward neural network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, seed=12345):
        super().__init__()
        torch.manual_seed(seed)
        #variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        #layers architecture
        self.linear_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_layer2 = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.linear_layer3 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.linear_layer4 = nn.Linear(self.hidden_size, 1)
        
    def forward(self, inputs):
        out = self.linear_layer1(inputs)
        out = nn.functional.relu(out)
        out = self.linear_layer2(out)
        out = nn.functional.relu(out)
        out = self.linear_layer3(out)
        out = nn.functional.relu(out)
        out = self.linear_layer4(out)
        out = torch.sigmoid(out)
        return out
    
    #mirror .predict_proba() and .predict() methods for models 1, 2, and 3
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor)
        class1_probs = self.forward(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs, class1_probs))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)




#2 hidden layers
class FFNNA(nn.Module):
    def __init__(self, input_size, hidden_size, seed=12345):
        super().__init__()
        torch.manual_seed(seed)
        #variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        #layers architecture
        self.linear_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer3 = nn.Linear(self.hidden_size, 1)
        
    def forward(self, inputs):
        out = self.linear_layer1(inputs)
        out = nn.functional.relu(out)
        out = self.linear_layer2(out)
        out = nn.functional.relu(out)
        out = self.linear_layer3(out)
        out = torch.sigmoid(out)
        return out
    
    #mirror .predict_proba() and .predict() methods for models 1, 2, and 3
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor)
        class1_probs = self.forward(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs, class1_probs))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


#3 hidden layers
class FFNNB(nn.Module):
    def __init__(self, input_size, hidden_size, seed=12345):
        super().__init__()
        torch.manual_seed(seed)
        #variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        #layers architecture
        self.linear_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer4 = nn.Linear(self.hidden_size, 1)
        
    def forward(self, inputs):
        out = self.linear_layer1(inputs)
        out = nn.functional.relu(out)
        out = self.linear_layer2(out)
        out = nn.functional.relu(out)
        out = self.linear_layer3(out)
        out = nn.functional.relu(out)
        out = self.linear_layer4(out)
        out = torch.sigmoid(out)
        return out
    
    #mirror .predict_proba() and .predict() methods for models 1, 2, and 3
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor)
        class1_probs = self.forward(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs, class1_probs))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)



#4 hidden layers
class FFNNC(nn.Module):
    def __init__(self, input_size, hidden_size, seed=12345):
        super().__init__()
        torch.manual_seed(seed)
        #variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        #layers architecture
        self.linear_layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_layer5 = nn.Linear(self.hidden_size, 1)

    def forward(self, inputs):
        out = self.linear_layer1(inputs)
        out = nn.functional.relu(out)
        out = self.linear_layer2(out)
        out = nn.functional.relu(out)
        out = self.linear_layer3(out)
        out = nn.functional.relu(out)
        out = self.linear_layer4(out)
        out = nn.functional.relu(out)
        out = self.linear_layer5(out)
        out = torch.sigmoid(out)
        return out
    
    #mirror .predict_proba() and .predict() methods for models 1, 2, and 3
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor)
        class1_probs = self.forward(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs, class1_probs))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)





#logistic regression as neural network
class LogisticRegressionNN(nn.Module):
    def __init__(self, input_size, seed=12345):
        super().__init__()
        torch.manual_seed(seed)
        #variables
        self.input_size = input_size
        #layers
        self.linear_layer = nn.Linear(self.input_size, 1)
        
    def forward(self, inputs):
        out = self.linear_layer(inputs)
        out = torch.sigmoid(out)
        return out
    
    def predict_proba(self, X):
        X = torch.tensor(X).type(torch.FloatTensor)
        class1_probs = self.forward(X).detach().numpy()
        class0_probs = 1-class1_probs
        return np.hstack((class0_probs, class1_probs))
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


