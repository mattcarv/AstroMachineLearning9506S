# Due to an increasing amount of available astronomical data in surveys, the 
# classification of new sources has to be done using their imaging, which can be 
# misleading considering the complexity of astronomical objects. For a better 
# understanding, we can analyze their spectra carefully, accounting for emission 
# and absorption lines. Although it is a good alternative, this process takes a 
# lot of time.

# The aim of this Machine Learning project is to use raw photometry data (magnitudes 
# in different bands) from the SDSS to test if we can separate sources into their true
# classes without having to study their spectroscopic data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# DATA HANDLING ---------------------------------------------------------------

# The data consists on a .csv file with 100000 astronomical objects containing 
# the following parameters: Object Identifier, Right Ascension, Declination, 
# Magnitudes u, g, r, i and z, Run Number, Rerun Number, Camera Column, Field Number, 
# Object Identifier (for spectral data), Class, Redshift, Plate Number, MJD and Fiber Identifier.

# For this investigation, we will need the object classes, 
# their magnitudes (all five bands) and occasionally their redshifts.

# To test the data using redshift, make sure to delete it from 'filename.drop'.

# Data input-------------------------------------------------------------------
filename = pd.read_csv('C:/Users/mathe/Downloads/archive/star_classification.csv')
df = filename.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID',
              'fiber_ID', 'plate', 'MJD', 'spec_obj_ID', 'redshift'], axis = 1)

# Relabelling -----------------------------------------------------------------

# We have to turn our classes into something the code can understand better than
# strings, therefore we can assign a number to each class.

class_dict = {"GALAXY": 0, "STAR": 1, "QSO": 2}

df["class"] = np.array([class_dict.get(x, 3) for x in df["class"]])

# Data Visualization ----------------------------------------------------------

# Using the seaborn package, we can visualize how imbalanced the data is.

import seaborn as sns

sns.countplot(df["class"])
plt.xlabel('Class')
plt.ylabel('Number')
plt.title("Class ", fontsize=16)
plt.clf()

# Condition to drop one of the categories -------------------------------------

# In here, we can select a class to drop, this will be helpful to further analysis
# when we need to compare Stars and QSOs. To compare all three classes, coment out 
# the next block.

condition = df['class'] != 0
df = df[condition]
df.drop(df[condition].index)
# print(df)

# Reshaping imbalanced data ---------------------------------------------------

# Since we have a huge imbalance, we can use the SMOTE task of the imblearn package
# to oversample the classes with less objects, using this we can analyze the relationship
# between classes.

from imblearn.over_sampling import SMOTE
from collections import Counter

x = df.drop(['class'], axis = 1)
y = df.loc[:,'class'].values

over_sample = SMOTE(random_state=1)
x, y = over_sample.fit_resample(x, y)

print('Original distribution: %s' % Counter(y))
print('Resampled distribution (minority classes were boosted): %s' % Counter(y))

# Standardization of data -----------------------------------------------------

# Here we have to set a standard in our dataframe using a scaler from the 
# preprocessing subpackage of sklearn.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# In this work, we use a 80/20 split between training and testing sets. A random
# seed was also included so we can have the same piece of information being 
# used by the ML techniques.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state = 1)

# SUPERVISED METHODS ----------------------------------------------------------

# Now we can apply this data to supervised methods to infer if it's possible to 
# separate astronomical objects into classes based on their visual magnitudes.
# The metric in this case is the classification report, which will return us the
# macro average accuracy, and also precision, recall and f1-score. Those parameters
# are great to visualize the relation between true/false positives and true/false
# negatives.

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# SVM -------------------------------------------------------------------------
# First, using Support Vector Machine.

from sklearn import svm

svm_classifier = svm.SVC(kernel='rbf', random_state=1)
svm_classifier.fit(x_train,y_train)
predicted = svm_classifier.predict(x_test)


# We can plot a confusion matrix that shows how well are the objects being 
# classified based on their classes. A good confusion matrix consists in a high
# number of True positives and True negatives, as well as a low number of False
# positives and False negatives.


cm = confusion_matrix(y_test, predicted, labels=svm_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(classification_report(y_test, predicted))

# Random Forest ---------------------------------------------------------------
# Similarly, we apply the same dataframe to the Random Forest method

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(x_train,y_train)
predicted = rf_classifier.predict(x_test)


# The confusion matrix is used here again to understand the distribution of the
# classification report.

cm = confusion_matrix(y_test, predicted, labels=rf_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(classification_report(y_test, predicted))

# UNSUPERVISED METHODS --------------------------------------------------------

# We can also apply unsupervised methods to separate classes and see how well they
# can perform under different methods. For each method, the metric of 'how good
# it works' is the Silhouette Score after applying clustering techniques to the
# respective distribution.

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture

# PCA with magnitudes ---------------------------------------------------------
# For the first unsupervised method (Principal Component Analysis), we will use
# the original standardized dataframe without the classes to test the class
# separation.

from sklearn.decomposition import PCA

df_noclasses = pd.DataFrame(x, y)
print(df_noclasses)

pca = PCA(n_components=2)

pca.fit(df_noclasses)

print("Explained variance ratio: ")
print(pca.explained_variance_ratio_)

print('PCA components: ')
print(pca.components_)

df_pca = pca.transform(df_noclasses)


#------------------------------------------------------------------------------
# With the PCA already transformed and fitted, we use clustering techniques to 
# separate regions in the relation of the first two components of the analysis.

# For this, we apply KMeans (with the 'k-means++' initial parameters), 
# Bayesian Gaussian Mixture (with a spherical covariance type) and a simple 
# DBSCAN analysis. Those techniques were chosen due to the size of our dataset
# and also because we could manually set the amount of clusters that we were 
# expecting.
#------------------------------------------------------------------------------


# kmeans = KMeans(n_clusters=2, init='k-means++')
# kmeans.fit(df_pca)
# labels = kmeans.labels_


gm = BayesianGaussianMixture(n_components=2, random_state=1, 
                              covariance_type='spherical', 
                              init_params='random').fit(df_pca)
labels = gm.predict(df_pca)
gm.means_


# db = DBSCAN(eps=4, min_samples=5).fit(df_pca)
# labels = db.labels_


# Setting plotting limits so we don't cluster outliers.

limits = [-0.6, 0.2, 0, 7]

plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5, c=labels)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.axis(limits)
plt.clf()

from sklearn.metrics import silhouette_score

silhouette_score = silhouette_score(df_noclasses, labels, metric='euclidean')
print(silhouette_score)


# t-SNE with magnitudes------------------------------------------------------
# Similarly to the PCA, t-SNE is also an unsupervised machine learning method,
# but in this case we have a non-linear fitting process. We will use the same 
# dataset we studied in the PCA section.


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=70, n_iter=3000, verbose=True, 
            random_state=1)


# Here we can differentiate tests for the same data using distinct perplexities
# and number of iterations.

df_tsne = tsne.fit_transform(df_noclasses)


# Again, applying the same three clustering techniques to our two-component
# unsupervised fitting.


# kmeans = KMeans(n_clusters=2)
# kmeans.fit(df_tsne)
# labels = kmeans.labels_

# db = DBSCAN(eps=4, min_samples=5).fit(df_tsne)
# labels = db.labels_

# gm = BayesianGaussianMixture(n_components=2, random_state=1, covariance_type='spherical', 
#                      init_params='random').fit(df_tsne)
# labels = gm.predict(df_tsne)
# gm.means_


# Also for a large dataset, we can use the BIRCH clustering technique, based on
# euclidean distance between points.

from sklearn.cluster import Birch

birch = Birch(n_clusters=2).fit(df_tsne)
labels = birch.predict(df_tsne)

plt.scatter(df_tsne[:, 0], df_tsne[:, 1], alpha=0.5)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

silhouette_score = silhouette_score(df_noclasses, labels, metric='euclidean')
print(silhouette_score)
