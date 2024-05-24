# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:13:40 2024

@author: jarre
"""
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
def plot_permutation_importance(clf, X, y, ax):#,div=106):
    result = permutation_importance(clf, X, y, n_repeats=15, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()
    #divide = 106-div
    #ax.tick_params(axis='both', which='major', labelsize=10)
    #ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    return ax
def vote(data,labels,weights = [3,4,2,1]):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier,GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    slk = SelectKBest(chi2,k=23)
    dat = slk.fit_transform(data,labels)
    names = slk.get_feature_names_out()
    #print(f"get fet: {names}")
    #print(f"dr: {slk.feature_names_in_}")
    mdd = pd.DataFrame(dat)
    mdd.columns=names
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(mdd, labels, test_size=0.25, random_state=42, stratify=labels)
    
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
    voting_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = voting_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Ensemble Model vote Accuracy: {accuracy:.4f}')
    cm = confusion_matrix(y_test, y_pred,normalize='true')
    #print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="0.3f", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix eclf vote")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    # weightlist = []
    # for name, clf in voting_classifier.named_estimators_.items():
    #     clf.fit(X_train, y_train)
    #     y_pred_individual = clf.predict(X_test)
    #     accuracy_individual = accuracy_score(y_test, y_pred_individual)
    #     print(f'{name} Accuracy: {accuracy_individual:.4f}')
    #     print(classification_report(y_test, y_pred_individual))
    #     cm = confusion_matrix(y_test, y_pred_individual,normalize='true')
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt="f", cmap="Blues", cbar=False)
    #     plt.title("Confusion Matrix clf vote")
    #     plt.xlabel("Predicted Labels")
    #     plt.ylabel("True Labels")
    #     plt.show()
    #     weightlist.append(accuracy_individual)
    fig, ax = plt.subplots(figsize=(8, 6))
    cutoff=15
    plot_permutation_importance(voting_classifier, X_test, y_test, ax)
    ax.set_title(f"Permutation Importances on the KBest features\n(test set)")
    ax.set_xlabel("Decrease in accuracy score")
    _ = ax.figure.tight_layout()
    return voting_classifier

rng = np.random.RandomState(seed=42)
mdf = pd.read_csv("moutput.csv")
y = mdf["gtype"]
X = mdf.drop(["gtype","seed","parameter"],axis="columns")
#X["random_num"] = rng.randn(X.shape[0])

vote(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# clf = GradientBoostingClassifier(n_estimators=150,min_samples_leaf=20, max_leaf_nodes=15,random_state=42,verbose=1)

# #clf = DecisionTreeClassifier(criterion="log_loss",splitter="best",min_samples_split=2,min_samples_leaf=20, max_leaf_nodes=21,random_state=42)
# clf.fit(X_train, y_train)
# print(f"Baseline accuracy on test data: {clf.score(X_test, y_test):.2}")
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
# tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
# tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,16))
# mdi_importances.sort_values().plot.barh(ax=ax1)
# ax1.set_xlabel("Gini importance")
# plot_permutation_importance(clf, X_train, y_train, ax2)
# ax2.set_xlabel("Decrease in accuracy score")
# fig.suptitle(
#     "Impurity-based vs. permutation importances on multicollinear features (train set)"
# )
# _ = fig.tight_layout()

# fig, ax = plt.subplots(figsize=(7, 14))
# plot_permutation_importance(clf, X_test, y_test, ax)
# ax.set_title("Permutation Importances on multicollinear features\n(test set)")
# ax.set_xlabel("Decrease in accuracy score")
# _ = ax.figure.tight_layout()

# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,16))
# corr = spearmanr(X).correlation

# # Ensure the correlation matrix is symmetric
# corr = (corr + corr.T) / 2
# np.fill_diagonal(corr, 1)

# # We convert the correlation matrix to a distance matrix before performing
# # hierarchical clustering using Ward's linkage.
# distance_matrix = 1 - np.abs(corr)
# dist_linkage = hierarchy.ward(squareform(distance_matrix))
# dendro = hierarchy.dendrogram(
#     dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
# )
# dendro_idx = np.arange(0, len(dendro["ivl"]))

# ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
# ax2.set_yticklabels(dendro["ivl"])
# _ = fig.tight_layout()
# from collections import defaultdict

# cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
# cluster_id_to_feature_ids = defaultdict(list)
# for idx, cluster_id in enumerate(cluster_ids):
#     cluster_id_to_feature_ids[cluster_id].append(idx)
# selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# selected_features_names = X.columns[selected_features]

# X_train_sel = X_train[selected_features_names]
# X_test_sel = X_test[selected_features_names]

# #clf_sel = GradientBoostingClassifier(n_estimators=150,min_samples_leaf=20,max_leaf_nodes=15,random_state=42)

# clf_sel = DecisionTreeClassifier(min_samples_leaf=20, max_leaf_nodes=15,random_state=42)
# clf_sel.fit(X_train_sel, y_train)
# print(
#     "Baseline accuracy on test data with features removed:"
#     f" {clf_sel.score(X_test_sel, y_test):.2}"
# )
# fig, ax = plt.subplots(figsize=(7, 6))
# plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
# ax.set_title("Permutation Importances on selected subset of features\n(test set)")
# ax.set_xlabel("Decrease in accuracy score")
# ax.figure.tight_layout()
# plt.show()