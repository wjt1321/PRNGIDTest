# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:44:17 2024

@author: jarre
"""

import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis,RadiusNeighborsClassifier
#from sklearn.pipeline import Pipeline
#import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier,ExtraTreesClassifier
#from sklearn.model_selection import GridSearchCV, KFold
#import plotly.colors as colors
#import plotly.express as px
#from plotly.subplots import make_subplots
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


#### Global vars

#np.random.seed(31415)

#print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

#kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
#bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

#kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
#bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

#pca = PCA(n_components=n_digits).fit(data)
#print(pca.components_)
#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)
#kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
#bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)
#
#print(82 * "_")

    #### Global vars

    #np.random.seed(31415)
n_digits = 6
dfraw = pd.read_csv("output.csv")
#print(dfraw)
data = dfraw.drop(["0","1","2","109"],axis = "columns")
labels = dfraw["0"]
#print(data)
mapping = {'CMRG': 0, 'LCG': 1,'LCG64':2,'LFG':3,'MLFG':4,'PMLCG':5}
mlabels =labels.replace(mapping)
OMP_NUM_THREADS = 5
colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"orchid","LFG":"maroon","MLFG":"green","PMLCG":"saddlebrown"}



#Creates ellipses for data vis
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ellipse.set(linewidth=3)
    return ax.add_patch(ellipse)

#does kmeans and several summary statistics
def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))




# a kemans on pca data
def pcaKmeans(data, labels):
    trainx, testx, trainy, testy = train_test_split(data,labels,random_state=42,stratify=labels)
    reduced_data = PCA(n_components=2).fit_transform(trainx)
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
    kmeans.fit(reduced_data)
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on PRNG (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
#pcaKmeans()

#colormap = {"LFG":"maroon","MLFG":"Orchid"}
#knn of nca data, actually pretty good
def ncaknn(data,labels):
    #for i in range(2,11):
        #reduced_data = PCA(n_components=2).fit_transform(data)
        trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.2,random_state=42,stratify=labels)
        nca = NeighborhoodComponentsAnalysis()#n_components=2)
        knn =  KNeighborsClassifier(n_neighbors=7,weights="distance")        
        pipeline = Pipeline([
            ('nca', nca),
            ('knn', knn)
        ])
        pipeline.fit(trainx, trainy)
        accuracy = pipeline.score(testx, testy)
        print(f"Accuracy: {accuracy:.2f}")
    
        # Predictions on the test set
        y_pred = pipeline.predict(testx)
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(testy, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(testy, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix nca knn")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

#pca knn model, worse then nca version, kernal is flexible so you can try different algorithms on it        
def pcaknn(data,labels):
    #for i in range(2,11):
        #reduced_data = PCA(n_components=2).fit_transform(data)
        trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.2,random_state=42,stratify=labels)
        #pca = PCA()#n_components=2)
        pca = KernelPCA(
        kernel="rbf", fit_inverse_transform=True, alpha=0.1
        )
        knn = KNeighborsClassifier(n_neighbors=7,weights="distance")
        #knn = KNeighborsClassifier(weights="distance")
        
        
        
        pipeline = Pipeline([
            ('pca', pca),
            ('knn', knn)
        ])
        pipeline.fit(trainx, trainy)
        accuracy = pipeline.score(testx, testy)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Predictions on the test set
        y_pred = pipeline.predict(testx)
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(testy, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(testy, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix pca knn")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()
#combination of 15 decision trees into one.
def baggingtrees(data,labels):
    #for i in range(8,30):
        trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.2,random_state=42,stratify=labels)
        #y = pd.get_dummies(labels)
        #y.replace({True:1,False:0})
        
        n_samples, n_features = data.shape
        #N_CORES = joblib.cpu_count(only_physical_cores=True)
        clf = DecisionTreeClassifier(criterion="log_loss",splitter="best",min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=22)
        cle = BaggingClassifier(estimator=clf,n_estimators=15,oob_score=True,random_state=42)
        cle = cle.fit(trainx, trainy)
        ypreds = cle.predict(testx)
        accuracy = accuracy_score(testy, ypreds)
        print(cle.oob_score_)
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(testy, ypreds))
        cm = confusion_matrix(testy, ypreds,normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix bagging trees")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()
#very strong model
def dectrees(data,labels):
    #for i in range(4,31):
      trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.25,random_state=42,stratify=labels)
      X = data
      y = labels
      #y = pd.get_dummies(labels)
      #y.replace({True:1,False:0})
      n_samples, n_features = data.shape
      #N_CORES = joblib.cpu_count(only_physical_cores=True)
      #clf = DecisionTreeClassifier(criterion="entropy",splitter="best",min_samples_split=2,min_samples_leaf=5, max_leaf_nodes=22,random_state=42)
      clf = DecisionTreeClassifier(criterion="log_loss",splitter="best",max_depth=15,min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=22,random_state=42)
      clf = clf.fit(trainx, trainy)
      #clf = clf.fit(X, y)
      #print(clf.feature_importances_)
      #feat_import = clf.feature_importances_
      #std = np.std([.feature_importances_)
      fig, ax = plt.subplots()
      #feat_import.plot.bar(yerr=std, ax=ax)
      #ax.set_title("Feature importances from Decision Trees")
      #ax.set_ylabel("Mean decrease in impurity")
      #fig.tight_layout()
      ypreds = clf.predict(testx)
      accuracy = accuracy_score(testy, ypreds)
      #accuracy = cross_val_score(clf,X,y,cv=5)
      print(f"Accuracy: {accuracy.mean():.5f}")
      print("Classification Report:")
      #print(classification_report(testy, ypreds))
      cm = confusion_matrix(testy, ypreds)
      #plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
      plt.title("Confusion Matrix Decision Tree")
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.show()

#histogram gradient boosting is significantly faster then regular gradient boosting, almost identical results      
def histgrad(data,labels):
    #for i in range(50,60):
      #pca = NeighborhoodComponentsAnalysis()
      #pcdata = pca.fit_transform(data,labels)
      trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.2,random_state=42,stratify=labels)
      #y = pd.get_dummies(labels)
      #y.replace({True:1,False:0})
      n_samples, n_features = data.shape
      #N_CORES = joblib.cpu_count(only_physical_cores=True)
      clf = HistGradientBoostingClassifier(max_leaf_nodes=31,min_samples_leaf=20,interaction_cst="pairwise",l2_regularization=0.6,random_state=42,class_weight="balanced")
      #clf = clf.fit(data, labels)
      clf = clf.fit(trainx, trainy)
      #print(clf.feature_importances_)
      ypreds = clf.predict(testx)
      accuracy = accuracy_score(testy, ypreds)
      #accuracy = cross_val_score(clf,data,labels,cv=5)
      print(f" Accuracy: {accuracy:.4f}")
      print("Classification Report:")
      #ypredstrain = clf.predict(trainx)
      #print(classification_report(trainy,ypredstrain))
      print(classification_report(testy, ypreds))
      #print(confusion_matrix(trainy, ypredstrain))
      cm = confusion_matrix(testy, ypreds,normalize="true")
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
      plt.title("Confusion Matrix hist grad")
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.show()
# attempted gradient boosting from scikit learn and another library, booth performed the same
def gradboost(data, labels):
    #for i in range(8,21):
      import xgboost as xgb
      from sklearn.metrics import accuracy_score
      #pca = PCA()
      #pcdata = pca.fit_transform(data,labels)
      trainx, testx, trainy, testy = train_test_split(data,mlabels,test_size = 0.2,random_state=42,stratify=labels)
      #y = pd.get_dummies(labels)
      #y.replace({True:1,False:0})
      n_samples, n_features = data.shape
      #N_CORES = joblib.cpu_count(only_physical_cores=True)
      clf = xgb.XGBClassifier()
      #clf = clf.fit(data, labels)
      clf = clf.fit(trainx, trainy)
      #print(clf.feature_importances_)
      ypreds = clf.predict(testx)
      accuracy = accuracy_score(testy, ypreds)
      #accuracy = cross_val_score(clf,data,labels,cv=5)
      print(f" Accuracy: {accuracy:.4f}")
      print("Classification Report:")
      #ypredstrain = clf.predict(trainx)
      #print(classification_report(trainy,ypredstrain))
      print(classification_report(testy, ypreds))
      #print(confusion_matrix(trainy, ypredstrain))
      cm = confusion_matrix(testy, ypreds,normalize="true")
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
      plt.title("Confusion Matrix xgboost")
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.show()      

#support vector machine, tried various approaches, none worked that well.      
def svmc(data,labels):    
   #for i in range(1,8):
      trainx, testx, trainy, testy = train_test_split(data,labels,test_size = 0.2,random_state=42,stratify=labels)
      #y = pd.get_dummies(labels)
      #y.replace({True:1,False:0})
      #SelectKBest(f_classif, k=10).fit_transform(trainx, trainy)
      nca = NeighborhoodComponentsAnalysis()
      n_samples, n_features = data.shape
      #N_CORES = joblib.cpu_count(only_physical_cores=True)
      scaler = StandardScaler()
      svc = SVC(kernel="rbf",class_weight="balanced")
      #skb = SelectKBest(chi2, k=23)#n_components=2)
      #LDA = LinearDiscriminantAnalysis()
     #knn = KNeighborsClassifier(n_neighbors=14)
      
      pipeline = Pipeline([
          #('scaler', scaler),
          #("skb",skb),
          ('nca',nca),
          ('svc', svc)
      ])
      pipeline.fit(trainx, trainy)
      accuracy = pipeline.score(testx, testy)
      print(f" Accuracy: {accuracy:.2f}")
      
      # Predictions on the test set
      y_pred = pipeline.predict(testx)
      print("Classification Report:")
      print(classification_report(testy, y_pred))
      cm = confusion_matrix(testy, y_pred,normalize="true")
      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
      plt.title("Confusion Matrix svmc grad")
      plt.xlabel("Predicted Labels")
      plt.ylabel("True Labels")
      plt.show()

#I tried to create a neat visualization with this code super late one night, it does not work and I forgot what it was
def transfromations():
    df = dfraw.drop(["1","2","109"],axis="columns")
    #df+=1
    pca = PCA(n_components=1)
    mpca = pca.fit(data)
    #N_SAMPLES = 1000
    FONT_SIZE = 6
    BINS = 30
    
    rng = np.random.RandomState(304)
    bc = PowerTransformer(method="box-cox")
    yj = PowerTransformer(method="yeo-johnson")
    # n_quantiles is set to the training set size rather than the default value
    # to avoid a warning being raised by this example
    qt = QuantileTransformer(
        n_quantiles=500, output_distribution="normal", random_state=rng
    )

    # create plots
    distributions = [
        ("LFG", (df[df["0"]=="LFG"].drop("0",axis="columns"))),
        ("MLFG", (df[df["0"]=="MLFG"].drop("0",axis="columns"))),
        ("LCG", (df[df["0"]=="LCG"].drop("0",axis="columns"))),
        ("LCG64", (df[df["0"]=="LCG64"].drop("0",axis="columns"))),
        ("CMRG", (df[df["0"]=="CMRG"].drop("0",axis="columns"))),
        ("PMLCG", (df[df["0"]=="PMLCG"].drop("0",axis="columns")))
    ]

    colors = ["#D81B60", "#0188FF", "#FFC107", "#B7A2FF", "#000000", "#2EC5AC"]

    fig, axes = plt.subplots(nrows=8, ncols=3, figsize=plt.figaspect(2))
    axes = axes.flatten()
    axes_idxs = [
        (0, 3, 6, 9),
        (1, 4, 7, 10),
        (2, 5, 8, 11),
        (12, 15, 18, 21),
        (13, 16, 19, 22),
        (14, 17, 20, 23),
    ]
    axes_list = [(axes[i], axes[j], axes[k], axes[l]) for (i, j, k, l) in axes_idxs]


    for distribution, color, axes in zip(distributions, colors, axes_list):
        name, X = distribution
        mX = mpca.transform(X)
        #print(mX)
        X_train, X_test = train_test_split(mX, test_size=0.5)
        # perform power transforms and quantile transform
        X_trans_bc = bc.fit(X_train+2).transform(X_test+2)
        lmbda_bc = round(bc.lambdas_[0], 2)
        X_trans_yj = yj.fit(X_train).transform(X_test)
        lmbda_yj = round(yj.lambdas_[0], 2)
        X_trans_qt = qt.fit(X_train).transform(X_test)

        ax_original, ax_bc, ax_yj, ax_qt = axes
        
        ax_original.hist(X_train, color=color, bins=BINS)
        ax_original.set_title(name, fontsize=FONT_SIZE)
        ax_original.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

        for ax, X_trans, meth_name, lmbda in zip(
            (ax_bc, ax_yj, ax_qt),
            (X_trans_bc, X_trans_yj, X_trans_qt),
            ("Box-Cox", "Yeo-Johnson", "Quantile transform"),
            (lmbda_bc, lmbda_yj, None),
        ):
            ax.hist(X_trans, color=color, bins=BINS)
            title = "After {}".format(meth_name)
            if lmbda is not None:
                title += "\n$\\lambda$ = {}".format(lmbda)
            ax.set_title(title, fontsize=FONT_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
            ax.set_xlim([-3.5, 3.5])


    plt.tight_layout()
    plt.show()

#this is a visualization of pca data
def pcagraph(dfraw,labels):
    #x = dfraw[(dfraw["0"]=="LFG" )|( dfraw["0"]=="MLFG")]
    x = dfraw
    #labelFam = {"CMRG":"CMRG","LCG":"LCG","LCG64":"LCG64","LFG":"LFG","MLFG":"LFG","PMLCG":"PMLCG"}
    # Split the data into training and testing sets
    #x = x[~((x['0'] != 'LFG') & (x["2"]!=1))]
    y = labels #labels.replace(labelFam)
    
    #print(x)
    x = x.drop(["0","1","2","109"],axis = "columns")
    #subset_df = train_df[train_df[column_name] == labels[lab]]
    #y = labels
    
    xdat = SelectKBest(chi2,k=23).fit_transform(x,y)
    
    # Apply PCA
    #pca = LinearDiscriminantAnalysis(solver="svd",n_components=2) 
    pca = PCA(n_components=2) 
    #pca = KernelPCA(
    #n_components=2, kernel="rbf", fit_inverse_transform=False, alpha=0.1
    #)
    principal_components = pca.fit_transform(xdat,y)  
    #nca = NeighborhoodComponentsAnalysis(n_components=2)
    #principal_components = nca.fit_transform(x.iloc[:, :-1],y)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])#,"PC3","PC4"])
    pc_df['label'] = y
    #print(pca.explained_variance_ratio_)
    #colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"green","LFG":"red","MLFG":"purple","PMLCG":"brown"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in pc_df['label'].unique():
        subset = pc_df[pc_df['label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'],c=colormaps[label],label=label)
        confidence_ellipse(subset['PC1'], subset['PC2'], ax,alpha=1, edgecolor=colormaps[label], zorder=0)#, kwargs)
    
    plt.title('LDA of PRNG Data colored by Generator')
    plt.xlabel(f'Component 1 (explained var) {pca.explained_variance_ratio_[0]*100:.1f}%')
    plt.ylabel(f'Component 2 (explained var) {pca.explained_variance_ratio_[1]*100:.1f}%')
    plt.legend()
    plt.show()
def ldagraph(dfraw,labels):
    #x = dfraw[(dfraw["0"]=="LFG" )|( dfraw["0"]=="MLFG")]
    x = dfraw
    #labelFam = {"CMRG":"CMRG","LCG":"LCG","LCG64":"LCG64","LFG":"LFG","MLFG":"LFG","PMLCG":"PMLCG"}
    # Split the data into training and testing sets
    #x = x[~((x['0'] != 'LFG') & (x["2"]!=1))]
    y = labels #labels.replace(labelFam)
    
    #print(x)
    x = x.drop(["0","1","2","109"],axis = "columns")
    #subset_df = train_df[train_df[column_name] == labels[lab]]
    #y = labels
    
    xdat = SelectKBest(chi2,k=23).fit_transform(x,y)
    
    # Apply PCA
    pca = LinearDiscriminantAnalysis(solver="svd",n_components=2) 
    #pca = PCA(n_components=2) 
    #pca = KernelPCA(
    #n_components=2, kernel="rbf", fit_inverse_transform=False, alpha=0.1
    #)
    principal_components = pca.fit_transform(xdat,y)  
    #nca = NeighborhoodComponentsAnalysis(n_components=2)
    #principal_components = nca.fit_transform(x.iloc[:, :-1],y)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])#,"PC3","PC4"])
    pc_df['label'] = y
    #print(pca.explained_variance_ratio_)
    #colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"green","LFG":"red","MLFG":"purple","PMLCG":"brown"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in pc_df['label'].unique():
        subset = pc_df[pc_df['label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'],c=colormaps[label],label=label)
        confidence_ellipse(subset['PC1'], subset['PC2'], ax,alpha=1, edgecolor=colormaps[label], zorder=0)#, kwargs)
    
    plt.title('LDA of PRNG Data colored by Generator')
    plt.xlabel(f'Component 1 (explained var) {pca.explained_variance_ratio_[0]*100:.1f}%')
    plt.ylabel(f'Component 2 (explained var) {pca.explained_variance_ratio_[1]*100:.1f}%')
    plt.legend()
    plt.show()
# we wanted to veiw a 3d version, it is a mess 
def pcagraph3d(dfraw,labes):
    #x = dfraw[(dfraw["0"]=="LFG" )|( dfraw["0"]=="MLFG")]
    x = dfraw
    #labelFam = {"CMRG":"CMRG","LCG":"LCG","LCG64":"LCG64","LFG":"LFG","MLFG":"LFG","PMLCG":"PMLCG"}
    # Split the data into training and testing sets
    #x = x[~((x['0'] != 'LFG') & (x["2"]!=1))]
    y = labels #labels.replace(labelFam)
    
    #print(x)
    x = x.drop(["0","1","2","109"],axis = "columns")
    #subset_df = train_df[train_df[column_name] == labels[lab]]
    #y = labels
    
    xdat = SelectKBest(chi2,k=23).fit_transform(x,y)
    
    # Apply PCA
    #pca = LinearDiscriminantAnalysis(solver="svd",n_components=3) 
    pca = PCA(n_components=3) 
    #pca = KernelPCA(
    #n_components=2, kernel="rbf", fit_inverse_transform=False, alpha=0.1
    #)
    principal_components = pca.fit_transform(xdat,y)  
    #nca = NeighborhoodComponentsAnalysis(n_components=2)
    #principal_components = nca.fit_transform(x.iloc[:, :-1],y)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2','PC3'])#,"PC3","PC4"])
    pc_df['label'] = y
    #print(pca.explained_variance_ratio_)
    #colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"green","LFG":"red","MLFG":"purple","PMLCG":"brown"}
    #fig, ax = plt.subplots(figsize=(8, 6))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for label in pc_df['label'].unique():
        subset = pc_df[pc_df['label'] == label]
        ax.scatter(subset['PC1'], subset['PC2'],subset['PC3'],c=colormaps[label],label=label)
        #confidence_ellipse(subset['PC1'], subset['PC2'], ax,alpha=1, edgecolor=colormaps[label], zorder=0)#, kwargs)
    
    plt.title('LDA of PRNG Data colored by Generator')
    ax.set_xlabel(f'Component 1 (explained var) {pca.explained_variance_ratio_[0]*100:.1f}%')
    ax.set_ylabel(f'Component 2 (explained var) {pca.explained_variance_ratio_[1]*100:.1f}%')
    ax.set_zlabel(f'Component 3 (explained var) {pca.explained_variance_ratio_[2]*100:.1f}%')
    plt.legend()
    plt.show()
    # plt.figure(figsize=(8, 6))
    # for label in pc_df['label'].unique():
    #     subset = pc_df[pc_df['label'] == label]
    #     plt.scatter(subset['PC3'], subset['PC4'], label=label)
    
    # plt.title('PCA of PRNG Data colored by Generator')
    # plt.xlabel(f'Principal Component 1 (PC3) {pca.explained_variance_ratio_[2]*100:.1f}%')
    # plt.ylabel(f'Principal Component 2 (PC4) {pca.explained_variance_ratio_[3]*100:.1f}%')
    # plt.legend()
    # plt.show()
    
#nca graph of first 2 components
def ncagraph(dfraw,y):
    #x = dfraw[(dfraw["0"]=="LFG" )|( dfraw["0"]=="MLFG")]
    x = dfraw
    #x = x[(x['0'] != 'LFG') & (x["2"]!=1)]

    y = x["0"]
    x = x.drop(["0","1","2","109"],axis = "columns")
    #y = labels
    
    xdat = SelectKBest(chi2,k=23).fit_transform(x,y)
    
    # Apply PCA
    nca = NeighborhoodComponentsAnalysis(n_components=4)
    principal_components = nca.fit_transform(xdat,y)

    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2',"PC3","PC4"])
    pc_df['label'] = y
    
    # Plot the data with colored points based on the labels
    #colormaps = {"CMRG":"blue","LCG":"orange","LCG64":"green","LFG":"red","MLFG":"purple","PMLCG":"brown"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in pc_df['label'].unique():
        subset = pc_df[pc_df['label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'],c=colormaps[label] ,label=label)
        confidence_ellipse(subset['PC1'], subset['PC2'], ax,alpha=0.8, edgecolor=colormaps[label], zorder=0)#, kwargs)
    
    plt.title('NCA of PRNG Data')
    plt.xlabel(f'Principal Component 1 (PC1)')
    plt.ylabel(f'Principal Component 2 (PC2)')
    plt.legend()
    plt.show()
    # plt.figure(figsize=(8, 6))
    # for label in pc_df['label'].unique():
    #     subset = pc_df[pc_df['label'] == label]
    #     plt.scatter(subset['PC3'], subset['PC4'], label=label)
    
    # plt.title('NCA of PRNG Data')
    # plt.xlabel(f'Principal Component 1 (PC3)')
    # plt.ylabel(f'Principal Component 2 (PC4)')
    # plt.legend()
    # plt.show()

#this is the current best performing model    
def vote(data, labels,weights = [3,4,2,1]):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier,GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    dat = SelectKBest(chi2,k=23).fit_transform(data,labels)
    #dat = StandardScaler().fit_transform(datm,labels)#, fit_params)
    #labelFam = {"CMRG":"CMRG","LCG":"LCG","LCG64":"LCG64","LFG":"COMBINED_LFG","MLFG":"COMBINED_LFG","PMLCG":"PMLCG"}
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dat, labels, test_size=0.25, random_state=42, stratify=labels)
    #X_train, X_test, y_train, y_test = train_test_split(dat, labels, test_size=0.25, random_state=42, stratify=labels)
    
    # Define individual classifiers
    hist_gradient_boosting = HistGradientBoostingClassifier(max_leaf_nodes=31,min_samples_leaf=20,interaction_cst="pairwise",l2_regularization=0.6,random_state=42,class_weight="balanced")
    gradient_boosting = GradientBoostingClassifier(n_estimators=150,min_samples_leaf=20, max_leaf_nodes=15,random_state=42,verbose=1)
    #ada = AdaBoostClassifier(algorithm = 'SAMME',n_estimators=100,random_state=42)
    random_forest = RandomForestClassifier(random_state=42,class_weight="balanced")
    decision_tree = DecisionTreeClassifier(criterion="log_loss",splitter="best",min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=21,class_weight="balanced")
    baggingClassifier = BaggingClassifier(estimator=decision_tree,n_estimators=15,random_state=42)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(weights="distance",n_neighbors=7)
    # Create a Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=[
            ('hist_gradient_boosting', hist_gradient_boosting),
            #('gradient_boosting', gradient_boosting),
            #('ada_boosting', ada),
            ('Bagging_Class', baggingClassifier),
            ('random_forests',random_forest),
            ('nca_knn', Pipeline([('nca', nca), ('knn', knn)]))],
        voting='soft',weights=weights  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
        )
    # Train the Voting Classifier
    #print(cross_val_score(voting_classifier, dat,labels.replace(labelFam),cv=5))
    
    voting_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = voting_classifier.predict(X_test)
    #unique_labs = ["CMRG","LCG","LCG64","LFG","MLFG","PMLCG"]
    unique_labs = ["CMRG","LCG","LCG64","COMBINED_LFG","PMLCG"]

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Ensemble Model vote Accuracy: {accuracy:.4f}')
    
    print(classification_report(y_test, y_pred))
    #mapp = {"CMRG":0,"LCG":1,"LCG64":2,"LFG":3,"MLFG":4,"PMLCG":5}
    
    #print(roc_auc_score(pd.Series(y_test).map(mapp),pd.Series(y_pred).map(mapp),multi_class="ovo",labels=unique_labs))
    cm = confusion_matrix(y_test, y_pred,normalize='true')
    #print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="0.3f", cmap="Blues", cbar=False,xticklabels=unique_labs,yticklabels=unique_labs)
    plt.title("Confusion Matrix of Model Predictions (test set) Normalized by Row")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    weightlist = []
    for name, clf in voting_classifier.named_estimators_.items():
        clf.fit(X_train, y_train)
        y_pred_individual = clf.predict(X_test)
        accuracy_individual = accuracy_score(y_test, y_pred_individual)
        print(f'{name} Accuracy: {accuracy_individual:.4f}')
        print(classification_report(y_test, y_pred_individual))
        cm = confusion_matrix(y_test, y_pred_individual,normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="0.3f", cmap="Blues", cbar=False,xticklabels=unique_labs,yticklabels=unique_labs)
        plt.title(f"Confusion Matrix {name} individual")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()
        weightlist.append(accuracy_individual)
    return smallest_to_largest(weightlist)

#slightly worse model then vote, was still working on this when the semester ended.
def stack(data,labels):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier,GradientBoostingClassifier,StackingClassifier
    from sklearn.tree import DecisionTreeClassifier

    
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Define individual classifiers
    hist_gradient_boosting = HistGradientBoostingClassifier(min_samples_leaf=20, max_leaf_nodes=15,random_state=42)
    #gradient_boosting = GradientBoostingClassifier(min_samples_leaf=20, max_leaf_nodes=15,verbose=1,random_state=42)
    #ada = AdaBoostClassifier(algorithm = 'SAMME',n_estimators=100,random_state=42)
    random_forest = RandomForestClassifier(min_samples_leaf=20, max_leaf_nodes=15,random_state=42)
    decision_tree = DecisionTreeClassifier(criterion="log_loss",splitter="best",min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=21,random_state=42)
    #nca = NeighborhoodComponentsAnalysis(random_state=42)
    #knn = KNeighborsClassifier(n_neighbors=7)
    # Create a Voting Classifier
    stacking_classifier = StackingClassifier(
        estimators=[
            ('hist_gradient_boosting', hist_gradient_boosting),
            #('gradient_boosting', gradient_boosting),
            #('ada_boosting', ada),
            ('decision_tree', decision_tree),
            ('random_forests',random_forest)]#,
            #('nca_knn', Pipeline([('nca', nca), ('knn', knn)]))]# 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
        )
    # Train the Voting Classifier
    stacking_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = stacking_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Ensemble Model stack Accuracy: {accuracy:.4f}')
    cm = confusion_matrix(y_test, y_pred,normalize='true')
    #print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix eclf stack")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    for name, clf in stacking_classifier.named_estimators_.items():
        clf.fit(X_train, y_train)
        y_pred_individual = clf.predict(X_test)
        accuracy_individual = accuracy_score(y_test, y_pred_individual)
        print(f'{name} Accuracy: {accuracy_individual:.4f}')
        print(classification_report(y_test, y_pred_individual))
        cm = confusion_matrix(y_test, y_pred_individual,normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix clf stack")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

#used to autocalc the weights for the voting classifier
def smallest_to_largest(ls):
    indexes = []
    for i in range(0,len(ls)):
        count = 0
        for j in range(0,len(ls)):
            if ls[i]>ls[j]:
                count+=1
        indexes.append(count+1)
        
    return indexes


#wanted this curve for my research poster, does not work atm
def rocauc(datad,labels):
    import matplotlib.pyplot as plt
    
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import RocCurveDisplay, auc
    from sklearn.model_selection import StratifiedKFold
    
    n_splits = 6
    cv = StratifiedKFold(n_splits=n_splits)
    #nca = NeighborhoodComponentsAnalysis()
    n_samples, n_features = data.shape
    #N_CORES = joblib.cpu_count(only_physical_cores=True)
    #scaler = StandardScaler()
    #svc = SVC(kernel="rbf",class_weight="balanced")
    #skb = SelectKBest(chi2, k=15)#n_components=2)
   #knn = KNeighborsClassifier(n_neighbors=14)
    hist_gradient_boosting = HistGradientBoostingClassifier(max_leaf_nodes=31,min_samples_leaf=20,interaction_cst="pairwise",l2_regularization=0.6,random_state=42,class_weight="balanced")
    #gradient_boosting = GradientBoostingClassifier(n_estimators=150,min_samples_leaf=20, max_leaf_nodes=15,random_state=42,verbose=1)
    #ada = AdaBoostClassifier(algorithm = 'SAMME',n_estimators=100,random_state=42)
    random_forest = RandomForestClassifier(random_state=42,class_weight="balanced")
    decision_tree = DecisionTreeClassifier(criterion="log_loss",splitter="best",min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=21,class_weight="balanced")
    baggingClassifier = BaggingClassifier(estimator=decision_tree,n_estimators=15,random_state=42)
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(weights="distance",n_neighbors=7)
    # Create a Voting Classifier
    voting_classifier = VotingClassifier(
        estimators=[
            ('hist_gradient_boosting', hist_gradient_boosting),
            #('gradient_boosting', gradient_boosting),
            #('ada_boosting', ada),
            ('Bagging_Class', baggingClassifier),
            ('random_forests',random_forest),
            ('nca_knn', Pipeline([('nca', nca), ('knn', knn)]))],
        voting='soft',weights=[3,4,2,1]  # 'hard' for majority voting, 'soft' for weighted voting based on class probabilities
        )
    classifier = voting_classifier
    X = datad.values
    y = labels.labels
    #X = nca.fit_transform(X,y)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n(Positive label 'Survived')",
    )
    ax.legend(loc="lower right")
    plt.show()

def main():
    #reduced_data = PCA(n_components=2).fit_transform(data)
    #kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
    #kmeans.fit(reduced_data)
    #print(PCA.explained_variance_ratio_)
    print(82 * "_")
    #gradboost()
    #transfromations()
    #stack()
    #print(82 * "_")
    #pcaKmeans(data,labels)
    #print(82 * "_")
    #pcagraph()
    
    weight1 = vote(data,labels)
    #print(82 * "_")
    #vote(weight1)
    #print(82 * "_")
    #pcagraph(dfraw,labels)
    #ncagraph(dfraw,labels)
    #rocauc(data,labels)
    #histgrad(data,labels)
    #svmc(data,labels)      
    #ncaknn(data,labels)
    #pcaknn(data,labels)
    #gradboost(data,labels)
    #baggingtrees(data,labels)
    #dectrees(data,labels) 
if __name__ == "__main__":
    main()