import numpy as np
from scipy.spatial.distance import euclidean,mahalanobis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



file = open("german.data-numeric", "r")             #read the file
data=np.zeros((1000,25))
for i,line in enumerate(file):                      #storing the data in numpy array
    for j,x in enumerate(line.split()):
        data[i,j]=x

result={}                                       #store actual class
label={}                                        #store predicted class
points=[]



for i in range(data.shape[0]):                  #storing the class of each data point and converting the points to tuple
    x=data[i].tolist()
    result[tuple(x[:-1])]=x[-1]
    points.append(x[:-1])

scaler.fit(points)
pp = scaler.transform(points)

data_T = np.array(points).T
data_cov = np.cov(data_T)

def dist(A,B):                                      #distance function
    return mahalanobis(A,B,data_cov)


def find_Neighbours(points,pt,eps):                 #function to calculate neighbours
    neighbours=[]
    for x in points:
        if dist(pt,x)<=eps:
            neighbours.append(x)
    return neighbours


def DBSCAN(points, eps, minPts):                    #function to perform DBSCAN
    C=0

    for x in points:
        if tuple(x) in label:
            continue

        neighbours = find_Neighbours(points=points,pt=x,eps=eps)

        if len(neighbours) < minPts:
            label[tuple(x)] = -1
            continue

        C+=1
        label[tuple(x)] = C

        S = neighbours.copy()
        S.remove(x)

        for y in S:
            if (tuple(y) in label and label[tuple(y)]==-1):
                label[tuple(y)] = C
            if tuple(y) in label:
                continue
            label[tuple(y)] = C
            N = find_Neighbours(points=points,pt=y,eps=eps)

            if(len(N)>=minPts):
                for q in N:
                    if q not in S:
                        S.append(q)




DBSCAN(points,eps=90,minPts=15)



correct=0
tp=0
tn=0
fp=0
fn=0

for x in points:                                #Calculating Accuracy
    #print('result='+str(result[tuple(x)])+'     label='+str(label[tuple(x)]))
    if(result[tuple(x)]==1 and label[tuple(x)]==1):
        tp+=1
    if (result[tuple(x)] == 2 and label[tuple(x)] == 1):
        tn+=1
    if (result[tuple(x)] == 1 and label[tuple(x)] == -1):
        fp+=1
    if(result[tuple(x)]==2 and label[tuple(x)]==-1):
        fn+=1

print('tp='+str(tp)+'   tn='+str(tn)+'      fn='+str(fn)+'      fp='+str(fp))
print('Accuracy = '+str((tp+fn)/1000)+'     precision = '+str(fn/(fn+fp))+'     recall = '+str(fn/(tn+fn)))
