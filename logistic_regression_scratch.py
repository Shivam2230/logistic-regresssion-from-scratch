import numpy as np
import csv
import matplotlib.pyplot as plt

def normalized(data):
    return (data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0))

def cost_function(beta,normalized,y):
    res=sigmoid(beta,normalized)
    step1 = y * np.log(res)
    step2 = (1 - y) * np.log(1 - res) 
    final = -step1 - step2 
    return np.mean(final)

def sigmoid(beta,X):
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

def log_gradient(beta, X, y):
    first_calc = sigmoid(beta, X) - y.reshape(X.shape[0], -1) 
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 
	
def gradient_desc(X, y, beta, lr=.01, converge_change=.001):
    cost = cost_function(beta, X, y) 
    change_cost = 1
    num_iter = 1 
    while(change_cost > converge_change):
            old_cost = cost 
            beta = beta - (lr * log_gradient(beta, X, y)) 
            cost = cost_function(beta, X, y) 
            change_cost = old_cost - cost 
            num_iter += 1
    return beta, num_iter
	
if __name__=="__main__":
    with open("dataset1.csv",'r') as file:
        
        lines=csv.reader(file)
        dataset=list(lines)
        for i in range(len(dataset)):
            dataset[i]=[float(x) for x in dataset[i]]
        dataset=np.array(dataset)
        normalized=normalized(dataset[:,:-1])
        X=np.hstack((np.matrix(np.ones(normalized.shape[0])).T,normalized))
        beta=np.matrix(np.zeros(X.shape[1]))
        y=dataset[:,-1]
        beta, num_iter = gradient_desc(X, y, beta) 
        print("Estimated regression coefficients:", beta)
        print("No. of iterations:", num_iter)
        pred_prob = sigmoid(beta, X)
        y_pred=np.squeeze(np.where(pred_prob >= .5, 1, 0))
        print("Correctly predicted labels:", np.sum(y == y_pred))
        ##plotting results
        data_0=X[np.where(y==0.0)]
        data_1=X[np.where(y==1.0)]
        plt.scatter([data_0[:,1]],[data_0[:,2]],c='b',label='y = 0')
        plt.scatter([data_1[:,1]],[data_1[:,2]],c='r',label='y = 1')
        x1=np.arange(0,1,0.1)
        x2=-(beta[0,0]+beta[0,1]*x1)/beta[0,2]
        plt.plot(x1,x2,c='k')
        plt.show()
