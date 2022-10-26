import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
# %matplotlib inline
import numpy as np


df = pd.read_csv('./FallData_Two.csv',index_col=0)


df.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']


data = df[['0','1']]
target = df['22']
target = np.array(target)


std_data = StandardScaler().fit_transform(data)
std_df=pd.DataFrame(std_data,columns=['0','1'])


model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(std_df)
visualizer.show()

k=4

km = KMeans(n_clusters= k, random_state=42)
km.fit(std_df)

df['cluster'] = km.fit_predict(std_df)

plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, '0'], df.loc[df['cluster'] == i, '1'],
                label = 'cluster ' + str(i))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('0', size = 12)
plt.ylabel('1', size = 12)
plt.show()

k_labels = km.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels == k) * (target == t)) for t in np.unique(target)]
    k_labels_matched[k_labels == k] = np.unique(target)[np.argmax(match_nums)]

    cm = confusion_matrix(target, k_labels_matched)

    print(cm)

k=5

km = KMeans(n_clusters= k, random_state=42)
km.fit(std_df)

df['cluster'] = km.fit_predict(std_df)

plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, '0'], df.loc[df['cluster'] == i, '1'],
                label = 'cluster ' + str(i))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('0', size = 12)
plt.ylabel('1', size = 12)
plt.show()


k_labels = km.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels == k) * (target == t)) for t in np.unique(target)]
    k_labels_matched[k_labels == k] = np.unique(target)[np.argmax(match_nums)]

    cm = confusion_matrix(target, k_labels_matched)

    print(cm)

k=6

km = KMeans(n_clusters= k, random_state=42)
km.fit(std_df)

df['cluster'] = km.fit_predict(std_df)

plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, '0'], df.loc[df['cluster'] == i, '1'],
                label = 'cluster ' + str(i))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('0', size = 12)
plt.ylabel('1', size = 12)
plt.show()


k_labels = km.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels == k) * (target == t)) for t in np.unique(target)]
    k_labels_matched[k_labels == k] = np.unique(target)[np.argmax(match_nums)]

    cm = confusion_matrix(target, k_labels_matched)

    print(cm)

k=7

km = KMeans(n_clusters= k, random_state=42)
km.fit(std_df)

df['cluster'] = km.fit_predict(std_df)

plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, '0'], df.loc[df['cluster'] == i, '1'],
                label = 'cluster ' + str(i))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('0', size = 12)
plt.ylabel('1', size = 12)
plt.show()


k_labels = km.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels == k) * (target == t)) for t in np.unique(target)]
    k_labels_matched[k_labels == k] = np.unique(target)[np.argmax(match_nums)]

    cm = confusion_matrix(target, k_labels_matched)

    print(cm)

k=8

km = KMeans(n_clusters= k, random_state=42)
km.fit(std_df)

df['cluster'] = km.fit_predict(std_df)

plt.figure(figsize = (8, 8))

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, '0'], df.loc[df['cluster'] == i, '1'],
                label = 'cluster ' + str(i))

plt.legend()
plt.title('K = %d results'%k , size = 15)
plt.xlabel('0', size = 12)
plt.ylabel('1', size = 12)
plt.show()


k_labels = km.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

for k in np.unique(k_labels):
        # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels == k) * (target == t)) for t in np.unique(target)]
    k_labels_matched[k_labels == k] = np.unique(target)[np.argmax(match_nums)]

    cm = confusion_matrix(target, k_labels_matched)

    print(cm)

