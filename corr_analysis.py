# https://towardsdatascience.com/let-us-understand-the-correlation-matrix-and-covariance-matrix-d42e6b643c22

import numpy as np
# import plotly.plotly as py
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import pandas as pd
from plotnine import *

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) # remove any empty lines

#selecting only first 4 columns as they are the independent(X) variable
# any kind of feature selection or correlation analysis should be first done on these
iris_X = iris_df.iloc[:,[0,1,2,3]]

# let us now standardize the dataset
iris_X_std = StandardScaler().fit_transform(iris_X)

# covariance matrix on standardized data
mean_vec = np.mean(iris_X_std, axis=0)

arr = np.arange(9).reshape(3,3)
arr
arr.T.dot(arr)
(iris_X_std-mean_vec).T.dot(iris_X_std-mean_vec) / (iris_X_std.shape[0]-1)

cov_matrix = (iris_X_std - mean_vec).T.dot((iris_X_std - mean_vec)) / (iris_X_std.shape[0]-1)


mean_vec = np.mean(iris_X, axis=0)
cov_matrix = (iris_X - mean_vec).T.dot((iris_X - mean_vec)) / (iris_X.shape[0]-1)


print('Covariance matrix \n%s' %cov_matrix)
cov_matrix

# correlation matrix:
cor_matrix = np.corrcoef(iris_X_std.T)
print('Correlation matrix using standardized data\n%s' %cor_matrix)

cor_matrix2 = np.corrcoef(iris_X.T)
print('Correlation matrix using base unstandardized data \n%s' %cor_matrix2)


eig_vals1, eig_vecs1 = np.linalg.eig(cov_matrix)
print('Eigenvectors using covariance matrix \n%s' %eig_vecs1)
print('\nEigenvalues using covariance matrix \n%s' %eig_vals1)

eig_vals2, eig_vecs2 = np.linalg.eig(cor_matrix)
print('Eigenvectors using correlation matrix of standardized data \n%s' %eig_vecs2)
print('\nEigenvalues using correlation matrix of standardized data \n%s' %eig_vals2)

eig_vals3, eig_vecs3 = np.linalg.eig(cor_matrix2)
print('Eigenvectors using correlation matrix of unstandardized data \n%s' %eig_vecs3)
print('\nEigenvalues using correlation matrix of unstandardized data \n%s' %eig_vals3)



# PLOTTING THE EXPLAINED VARIANCE CHARTS

##################################
#USING THE COVARIANCE MATRIX
##################################
tot = sum(eig_vals1)
explained_variance1 = [(i / tot)*100 for i in sorted(eig_vals1, reverse=True)]
cum_explained_variance1 = np.cumsum(explained_variance1)


vardata1 = pd.DataFrame(zip(explained_variance1, cum_explained_variance1, list(range(1,5)), ['cumulative explained variance']*4 ),columns=['exp_var','cum_var','x','class'])

(ggplot(vardata1) 
    + geom_col(aes(x='x', y='exp_var'), fill='gray') 
    + geom_point(aes(x='x', y='cum_var'), color='red', size=2) 
    + geom_line(aes(x='x', y='cum_var', color='class'), size=1, alpha=1)
    + labs(x='',y='Explained variance in percent', title='Explained variance by different principal components using the covariance matrix', color='')
#    + theme_xkcd()
)

# trace3 = Bar(
#         x=['PC %s' %i for i in range(1,5)],
#         y=explained_variance1,
#         showlegend=False)
#
# trace4 = Scatter(
#         x=['PC %s' %i for i in range(1,5)], 
#         y=cum_explained_variance1,
#         name='cumulative explained variance')
#
# data2 = Data([trace3, trace4])
#
# layout2=Layout(
#         yaxis=YAxis(title='Explained variance in percent'),
#         title='Explained variance by different principal components using the covariance matrix')
#
# fig2 = Figure(data=data2, layout=layout2)
# py.iplot(fig2)

##################################
#USING CORRELATION MATRIX
##################################

tot = sum(eig_vals3)
explained_variance = [(i / tot)*100 for i in sorted(eig_vals3, reverse=True)]
cum_explained_variance = np.cumsum(explained_variance)

vardata = pd.DataFrame(zip(explained_variance, cum_explained_variance, list(range(1,5)), ['cumulative explained variance']*4 ),columns=['exp_var','cum_var','x','class'])

(ggplot(vardata1) 
    + geom_col(aes(x='x', y='exp_var'), fill='gray') 
    + geom_point(aes(x='x', y='cum_var'), color='red', size=2) 
    + geom_line(aes(x='x', y='cum_var', color='class'), size=1, alpha=1)
    + labs(x='',y='Explained variance in percent', title='Explained variance by different principal components using the covariance matrix', color='')
#    + theme_xkcd()
)

# trace1 = Bar(
#         x=['PC %s' %i for i in range(1,5)],
#         y=explained_variance,
#         showlegend=False)
#
# trace2 = Scatter(
#         x=['PC %s' %i for i in range(1,5)], 
#         y=cum_explained_variance,
#         name='cumulative explained variance')
#
# data = Data([trace1, trace2])
#
# layout=Layout(
#         yaxis=YAxis(title='Explained variance in percent'),
#         title='Explained variance by different principal components using the correlation matrix')
#
# fig = Figure(data=data, layout=layout)
# py.iplot(fig)



