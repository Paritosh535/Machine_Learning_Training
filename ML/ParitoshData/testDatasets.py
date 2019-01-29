from sklearn.datasets.samples_generator import make_blobs,make_moons,make_circles
from matplotlib import pyplot as plt
from pandas import DataFrame

# generate 2 classification dataset
X,y = make_blobs(100,centers=6,n_features=2)

# scatter plot, dots colored by class value
print(X[:,0])
print(X[:,1])
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green',3:'black',4:'orange',5:'purple'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
print("done")


X,y = make_moons(100,noise=0.1)

df= DataFrame(dict(x=X[:,0],y=X[:,1],label=y))
colors= {0:'red',1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key,group in grouped:
    group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,color=colors[key])
plt.show()

print("make moons done")


# make_circles
# generate 2d classification dataset
X, y = make_circles(n_samples=100, noise=0.05)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()
print("circles done")