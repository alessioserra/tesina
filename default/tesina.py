import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
from sklearn.svm.classes import SVC
from sklearn.datasets import load_wine
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection._split import train_test_split
from sklearn.model_selection._validation import cross_val_score

##################################################
### Methods used during the analysis #############
##################################################

def getLabelDistribution(y):
    one = 0
    two = 0
    three = 0   
    for label in y:
        if label == 0:
            one = one + 1
        elif label == 1 : 
            two = two + 1
        elif label == 2:
            three = three + 1
    labels = 'class_0', 'class_1', 'class_2'
    sizes = [one, two, three]
    colors = ['lightgreen', 'lightskyblue', 'orange']
    explode = (0.0, 0.0, 0.0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=85)
    plt.title("Class labels distibution")
    plt.axis('equal')
    plt.show()

def boxplots(X):
    green_diamond = dict(markerfacecolor='g', marker='D')
    for idx in range(1,13):
        plt.subplot(3,5,idx)
        plt.axis('off')
        plt.boxplot(X[:,(idx-1)], flierprops=green_diamond)
    plt.show()
    
def corr_matrix(X):
    X_corr = pd.DataFrame(X)
    corr = X_corr.corr()
    sns.heatmap(corr, annot = True)
    plt.show()
    
def show_PCA(X):
    pca= PCA(n_components=2)
    X_pca = pca.fit_transform(X, y)
    # Plot result
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y,s=10)
    plt.title('2D representation of PCA')
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.tight_layout()
    plt.show()
    return X_pca

def show_LDA(X):
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    # Plot result
    plt.scatter(X_lda[:,0], X_lda[:,1], c=y,s=10)
    plt.title('2D representation of LDA')
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.show()
    return X_lda
 
def getContourImage(model, X):   
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = y_pred.reshape(xx.shape)
    color = model.predict(X)
    plt.title("Class boundaries")
    plt.pcolormesh(xx, yy, Z)
    plt.scatter(X[:,0], X[:,1], c = color, s=15, edgecolors='k')
    plt.axis('off')
    plt.show()
        
##################################################
############## Read the dataset ##################
##################################################

# Download wine dataset
X, y = load_wine(return_X_y = True)

# Plot pie chart label distribution
getLabelDistribution(y)

# Standardize Dataset:
X = preprocessing.scale(X)

# BoxPlots:
boxplots(X)

# CORRELATION MATRIX
corr_matrix(X)

# PCA
X_pca = show_PCA(X)

# LDA
X_lda = show_LDA(X)

# Support Vector Machines
C = [0.001,0.01,1,10,100]
for c in C:
    svm = SVC(kernel='linear', C=c)
    print("Values of C = ",c)
    print("X_stand avg_accuracy_score",np.mean(cross_val_score(svm, X, y, cv=5, scoring="accuracy")))
    print("X_PCA avg_accuracy_score",np.mean(cross_val_score(svm, X_pca, y, cv=5, scoring="accuracy")))
    print("X_LDA avg_accuracy_score",np.mean(cross_val_score(svm, X_lda, y, cv=5, scoring="accuracy")),"\n")

X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(X_lda, y, test_size = 0.2) 
svm = SVC(kernel='linear',C=10)
svm.fit(X_lda_train, y_lda_train)
getContourImage(svm, X_lda_test)

# KNN
K = [ 1, 3, 5, 7]
for k in K:
    knn = KNeighborsClassifier(n_neighbors=k)
    print("Values of K = ",k)
    print("X_stand avg_accuracy_score",np.mean(cross_val_score(knn, X, y, cv=5, scoring="accuracy")))
    print("X_PCA avg_accuracy_score",np.mean(cross_val_score(knn, X_pca, y, cv=5, scoring="accuracy")))
    print("X_LDA avg_accuracy_score",np.mean(cross_val_score(knn, X_lda, y, cv=5, scoring="accuracy")),"\n")

X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(X_lda, y, test_size = 0.2) 
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_lda_train, y_lda_train)
getContourImage(knn, X_lda_test)

# Logistic Regression



