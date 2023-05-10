import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

# feature names
colnames=['sepal_length', 'sepal_width', 'petal_length', 'petal_width','label'] 

df = pd.read_csv("D:\MSCS_UTA_RA\Spring 2023\Machine Learning\Assignments and Projects\Project3_Report_1002063832\iris.data",names=colnames,header=None,index_col=False)

#Displaying the first 10 rows of iris data
df.head(10)

# setting the lables into numericals
df.replace("Iris-setosa",1,inplace=True)
df.replace("Iris-versicolor",2,inplace=True)
df.replace("Iris-virginica",3,inplace=True)

#df.corr().style.background_gradient(cmap="Greens") # this requires Jinja2 , i used thisin report but will comment it out to simpl out the code and dependencies
print(df.corr())

new_df = df.drop('label',axis=1)
new_df = new_df.drop('petal_width',axis=1)
#new_df.corr().style.background_gradient(cmap="Greens") # this requires Jinja2 , i used thisin report but will comment it out to simpl out the code and dependencies
print(new_df.corr())



#defining the x,y,a and b as 4 respective features
x = df['petal_length']
y = df['petal_width']
a = df['sepal_length']
b = df['sepal_width']
z = df['label']

#Choosing sepal_length and sepal_width for clustering beacuse of reason mentioned in report
X = np.column_stack((a,b,z))
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")

plt.scatter(X[ : , 0], X[ :, 1]) # plotting the visualization graph for selected feature
plt.show()
X_with_label = X



print(np.random.choice(X[ : , 0],2,replace=False))

Random_centers=3

#Choosing random points among the data as initial centriods
centers = X[np.random.choice(X.shape[0], Random_centers, replace=False)]
centers

#Plotting the initial centriods
plt.scatter(X[ : , 0], X[ :, 1])
for i in centers:
  plt.scatter(i[0],i[1],color="red")
plt.show()


#Euclidean distance function

def calculate_distance(a,b):
  """
  Takes two points and returns the euclidean distance between the 2 points
  """
  dis = math.sqrt((abs(a[0]-b[0]))**2 + (abs(a[1]-b[1]))**2)

  return dis

#As we choose to do 3 clusters, given the reason in report.
#green,blue and yello represent the cluster colors in visualization.
green = []
yellow = []
blue = []

#Below for loop will assign each point to nearst cluster
for i in X:
  
  a = calculate_distance(centers[0,:2],i[:2])
  b = calculate_distance(centers[1,:2],i[:2])
  c = calculate_distance(centers[2,:2],i[:2])

  res = min(a,b,c)
  i = i.tolist()
  if res ==a:
    plt.scatter(i[0],i[1],color="green")
    green.append(i)
  elif res == b:
    plt.scatter(i[0],i[1],color="blue")
    blue.append(i)
  elif res==c:
    plt.scatter(i[0],i[1],color="yellow")
    yellow.append(i)

for i in centers:
  plt.scatter(i[0],i[1],color="red")
plt.show()


#Below function will compute new centriods based on initial assignment
def compute_new_centroids(a,b,c):
  
  c1 = np.average(a,axis=0).tolist()
  c2 = np.average(b,axis=0).tolist()
  c3 = np.average(c,axis=0).tolist()
  centers = [c1,c2,c3]

  for i in centers:
    i[0]=round(i[0],4)
    i[1]=round(i[1],4)
    
  return centers


#This is the main kmeans function which will do the iterations until we reach a point where centriods do not change
#and have stable clusters
def kmeans(centers):
  flag=1
  centers=centers[:,:2]
  while(flag):
    green = []
    yellow = []
    blue = []
    
    for i in X:
      #print(centers[0],i[:2])
      a = calculate_distance(centers[0],i[:2])
      b = calculate_distance(centers[1],i[:2])
      c = calculate_distance(centers[2],i[:2])

      res = min(a,b,c)
      i = i.tolist()
      if res ==a:
        plt.scatter(i[0],i[1],color="green")
        green.append(i)
      elif res == b:
        plt.scatter(i[0],i[1],color="blue")
        blue.append(i)
      elif res==c:
        plt.scatter(i[0],i[1],color="yellow")
        yellow.append(i)

    green = np.array(green)
    blue = np.array(blue)
    yellow = np.array(yellow)

    
    for i in centers:
      plt.scatter(i[0],i[1],color="red")
    
    plt.show()
    
    prev_centers=centers
    
    centers = compute_new_centroids(green[:,:2],blue[:,:2],yellow[:,:2])
    count=0
    for i in range(0,3):
      if prev_centers[i][0] == centers[i][0]:
        if prev_centers[i][1]==centers[i][1]:
          count=count+1
    if count==3:
      flag=0

    print("Previous Centriods are : ",prev_centers,"\n Current Centriods are : ",centers)
  
kmeans(centers) 

#Choosing the number of clusters
 # the inbuilt kmeans is only used for elbow method to check how many cluster to choose.
from sklearn.cluster import KMeans as km
wcss = []
x = X[ :, :2]        
for i in range(1, 11):
    kmeans = km(n_clusters=i, init='k-means++', max_iter=300, n_init = 10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss,color="pink")
plt.plot(range(1,11),wcss,'g^',color="blue")

plt.title('The Elbow Methdod')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


def accuracy(green,blue,yellow):

  green = np.array(green)
  blue = np.array(blue)
  yellow = np.array(yellow)
  green_counter = Counter(green[:,2])
  blue_counter = Counter(blue[:,2])
  yellow_counter = Counter(yellow[:,2])

  print(green_counter,yellow_counter,blue_counter)
  pred = 0
  green_max = max(green_counter.values())

  pred +=green_max

  blue_max = max(blue_counter.values())

  pred +=blue_max

  yellow_max = max(yellow_counter.values())

  pred +=yellow_max

  return (pred/150)*100

print("the accuracy after evaluating the final clusters is : ",accuracy(green,blue,yellow))