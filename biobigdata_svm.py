import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc
# %matplotlib inline

df = pd.read_csv('./FallData_Two.csv',index_col=0)
df.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
data=df.drop(columns=['22'])

std_data = StandardScaler().fit_transform(data)
std_df=pd.DataFrame(std_data,columns=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21'])


X_train, X_test, y_train, y_test = train_test_split(std_df,df['22'], test_size=0.30, random_state=42)


arr = [0.01, 0.1, 1, 10, 100]
for i in arr:
    s = SVC(kernel='linear', C=i, random_state=42,probability=True)  # c와 gamma값을 조정하면서 정확도를 높여보세요
    s.fit(X_train, y_train)
    y_pred = s.predict(X_test)

    print("C = " ,i,"confuseion_matrix :\n",confusion_matrix(y_test, y_pred))
    print("C = ", i, "test_acc:\n",s.score(X_test,y_test))
    print("C = ", i, "classification_report:\n",classification_report(y_test, y_pred))

    n_classes = 1
    y_fake_test = y_test.values - 1
    y_pred = y_pred - 1

    labels = [0, 1]
    y_fake_test = label_binarize(y_fake_test, classes=labels)
    y_pred = label_binarize(y_pred, classes=labels)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_fake_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_fake_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
