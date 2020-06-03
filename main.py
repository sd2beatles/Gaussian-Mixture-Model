import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GMM:
    def __init__(self,k,max_iter=5):
        self.k=k
        self.max_iter=int(max_iter)
    
    def initialize(self,X):
        self.shape=X.shape
        #assign the number of rows and columns to n and m,respectively
        self.n,self.m=self.shape
        #assigin the inital probabilities of each class
        self.phi=np.full(shape=self.k,fill_value=1/self.k)
        self.weights=np.full(shape=self.shape,fill_value=1/self.k)
        
        #Randomly choose rows from the orinial data for k times. This represents mu with k*m matrix
        random_rows=np.random.randint(low=0,high=self.n,size=self.k)
        self.mu=[X[random_row,:] for random_row in random_rows]
        self.sigma=[np.cov(X.T) for _ in range(self.k)]
    
    def e_step(self,X):
        #E-step:update weights and phi holding mu and sigma
        self.weights=self.predict_prob(X)
        #self.phi is equal to weight mean for each class
        self.phi = self.weights.mean(axis=0)
        
    def m_step(self,X):
        for i in range(self.k):
            weight=self.weights[:,[i]]
            total_weight=weight.sum()
            self.mu[i]=(np.dot(X.T,weight)/total_weight).reshape(1,-1)
            temp=weight*(X-self.mu[i])
            self.sigma[i]=np.dot(temp.T,(X-self.mu[i]))/total_weight
      
        
    def fit(self,X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
    
    def predict_prob(self,X):
        likelihood=np.zeros((self.n,self.k))
        for i in range(self.k):
            #compute the proability of X given the parameters of k
            distribution=multivariate_normal(mean=self.mu[i].ravel(),cov=self.sigma[i])
            likelihood[:,i]=distribution.pdf(X)
        numerator=likelihood*self.phi
        denominator=numerator.sum(axis=1)[:,np.newaxis]
        weights=numerator/denominator
        return weights
    
    def predict(self,X):
        weights=self.predict_prob(X)
        return np.argmax(weights,axis=1)


def jitter(x):
    return x+np.random.uniform(low=-0.5,high=0.05,size=x.shape)


def plot_axis_pairs(X,axis_pairs,clusters,classes):
    n_row=len(axis_pairs)//2
    n_col=2
    plt.figure(figsize=(12,6))
    for index,(x_axis,y_axis) in enumerate(axis_pairs):
        plt.subplot(n_row,n_col,index+1)
        plt.title('GMM Clusters')
        plt.xlabel(iris.feature_names[x_axis])
        plt.ylabel(iris.feature_names[y_axis])
        plt.scatter(
             jitter(X[:,x_axis]),
             jitter(X[:,y_axis]),
             c=clusters,
             cmap=plt.cm.get_cmap('brg'),
             marker='x')
    plt.tight_layout()

def main():
    np.random.seed(42)
    gmm=GMM(k=3,max_iter=10)
    iris=load_iris()
    X=iris.data
    gmm.fit(X)
    plot_axis_pairs(
    X=X,
    axis_pairs=[ 
        (0,1), (2,3), 
        (0,2), (1,3) ],
    clusters=gmm.predict(X),
    classes=iris.target)
 
if __name__=='__main__':
  main()
  
   


