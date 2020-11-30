import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering  
import random
from collections import OrderedDict
        
data=np.loadtxt('covid-19.txt',delimiter='\n',dtype=np.str)
data=data[1:]

x_data=[]
for i in range(len(data)):
    temp=data[i].split("\t")[5:7]
    temp=list(map(float,temp))
    x_data.append(temp)

    
x_data=np.array(x_data,dtype=np.float32)   
scaler=MinMaxScaler()
norm_x=scaler.fit_transform(x_data)




class kmeans:

    def __init__(self,data,n):
        self.data=data
        self.n=n
        self.cluster=OrderedDict()
        
    def init_center(self):
        index=random.randint(0,self.n)
        index_list=[]

        for i in range(self.n):
            while index in index_list:
                index=random.randint(0,self.n)
                
            index_list.append(index)
            
            self.cluster[i]={'center':self.data[index],
                            'data':[]}
    
    def get_dis(a,b):
        distance=np.linalg.norm(a-b)
        return distance 
    
    
    def clustering(self):
        
        for i in range(len(self.data)):
            distance_for_vec=[]
            for j in range(len(self.cluster)):
                distance=np.linalg.norm(self.cluster[j]['center']-self.data[i],ord=2)
                distance_for_vec.append(distance)
            min_index=distance_for_vec.index(min(distance_for_vec))    
            self.cluster[min_index]['data'].append(self.data[i])


    def update(self):
        cluster_data=[]
        mean_of_each_cluster=[]
        for i in range(len(self.cluster)):
            cluster_data.append(self.cluster[i]['data'])
        
        for i in range(len(self.cluster)):
            mean_of_each_cluster.append(np.mean(cluster_data[i],axis=0))

        for i in range(len(self.cluster)):

            self.cluster[i]['center']=mean_of_each_cluster[i]    


    def update_until_end(self):
        
        prev_center=[]
        after_center=[]
        for i in range(len(self.cluster)):
            prev_center.append(self.cluster[i]['center'])
        
        self.update()
        for i in range(len(self.cluster)):
            after_center.append(self.cluster[i]['center'])

        if np.array_equal(prev_center,after_center):
            print("learning over")
            return
        else:
            self.clustering()
            self.update()
            self.update_until_end()



       


    def get_result(self,cluster):
        results=[]
        labels=[]

        for key,value in cluster.items():
            for item in value['data']:
                labels.append(key)
                results.append(item)

        return np.array(results),labels   


    def draw_graph(self,data,labels):
        plt.figure() 
        plt.scatter(data[:,0],data[:,1],c=labels,cmap='rainbow')
        plt.show()


    def fit(self):
        self.init_center()
        self.clustering()
        self.update_until_end() 
        print("fnished")                






def draw_graph_for_db(eps,min_samples,data):
    db=DBSCAN(eps=eps,min_samples=min_samples)
    db.fit(data)
    prediction=db.fit_predict(data)
    plt.scatter(data[:,0],data[:,1],c=prediction,cmap='Paired')
    plt.show()



def draw_graph_for_agg(n_clusters,data):
    agg=AgglomerativeClustering(n_clusters=n_clusters,linkage='complete')
    agg.fit(data)
    prediction=agg.fit_predict(data)
    plt.scatter(data[:,0],data[:,1],c=prediction,cmap='Paired')
    plt.show()

  

if __name__=='__main__':
    
    draw_graph_for_db(0.1,2,norm_x)
    draw_graph_for_agg(8,norm_x)
    
    k=kmeans(norm_x,8)
    k.fit()
    result,label=k.get_result(k.cluster)
    k.draw_graph(result,label)
    