#! /usr/bin/env python
import matplotlib.pyplot as plt
from quilt.data.examples import uciml_iris
from quilt.data.uciml import wine_quality
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
encoder= LabelEncoder()

iris=uciml_iris.tables.iris()
wine=wine_quality.tables.red()
wine=wine.rename(index=str, columns={"quality": "class"}) #keeping class column label the same across

def preprocessing(df):
    '''creates labels df and removes lables from original df'''
    df_labels=df[['class']]
    df=df.drop('class' ,1)
    return df,df_labels

def str_check(Y):
    ''' checks if labels are strings, and if they are they are encoded'''
    labels=Y['class'].values.tolist()
    str_check=list(map(lambda x: isinstance(x,str),labels))
    if all(str_check):
        Y['class']=encoder.fit_transform(Y[['class']])
    return Y

def split_data(X,Y):
    '''in case we need to transform labels later'''
    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, valid_index in split.split(X, Y['class']):
        X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]
        Y_train, Y_valid=Y.iloc[train_index], Y.iloc[valid_index]
    return X_train, X_valid,Y_train,Y_valid

class Classifier():
    def __init__(self,X_train,X_test,Y_train,Y_test,kfolds=5, neighbors=3):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train=Y_train.values.ravel()
        self.Y_test=Y_test.values.ravel()
        # self.Y_train=str_check(Y_train)
        # self.Y_valid=str_check(Y_valid)

        self.kfolds = int(kfolds) #number of splits for cross validation
        self.neighbors = np.arange(1,neighbors,1)

    def lr(self):
        print('using one predictor at a time:')
        predictor_acc=dict()
        for predictor in self.X_train.columns.values.tolist():
            model = LogisticRegression(random_state=42,multi_class='multinomial',solver='lbfgs',max_iter=4000).fit(self.X_train[[predictor]], self.Y_train)
            probabilities=model.predict_proba(self.X_test[[predictor]])
            prediction=model.predict(self.X_test[[predictor]])
            acc_score=accuracy_score(self.Y_test,prediction)
            conf_matrix=confusion_matrix(self.Y_test,prediction)
            predictor_acc[predictor]=acc_score
            ##future updates do ROC for each predictor
        predictor_score=sorted(predictor_acc.items(), key=lambda x:x[1])

        print('top two predictors: ')
        for i in predictor_score[-2:]:
            print(i[0],'accuracy:',i[1])
        print('\n')

        print('using all predictors:')
        model = LogisticRegression(random_state=42,multi_class='multinomial',solver='lbfgs',max_iter=4000).fit(self.X_train, self.Y_train)
        probabilities=model.predict_proba(self.X_test)
        prediction=model.predict(self.X_test)
        acc_score=accuracy_score(self.Y_test,prediction)
        conf_matrix=confusion_matrix(self.Y_test,prediction)
        print('accuracy %f'%acc_score)
        print('confusion matrix \n',conf_matrix,'\n')
        ##do ROC for all predictors
        #print(model.scores_)

    def lrcv(self):
        model = LogisticRegressionCV(cv=self.kfolds, random_state=42,multi_class='multinomial',max_iter=4000).fit(self.X_train, self.Y_train)
        probabilities=model.predict_proba(self.X_test)
        prediction=model.predict(self.X_test)
        acc_score=accuracy_score(self.Y_test,prediction)
        conf_matrix=confusion_matrix(self.Y_test,prediction)
        print('accuracy %f'%acc_score)
        print('confusion matrix \n',conf_matrix)
        #print(model.scores_)

    def knn(self):
        cv_acc=[]
        for k in self.neighbors:
            model= KNeighborsClassifier(n_neighbors=k)
            score=cross_val_score(model,self.X_train, self.Y_train, cv=10, scoring='accuracy')
            cv_acc.append(score.mean())
        MSE=[1-x for x in cv_acc]
        optimal_k=self.neighbors[MSE.index(min(MSE))]
        print('optimal k is:',optimal_k)
        optimal_model= KNeighborsClassifier(n_neighbors=optimal_k).fit(self.X_train,self.Y_train)
        probabilities=optimal_model.predict_proba(self.X_test)
        prediction=optimal_model.predict(self.X_test)
        acc_score=accuracy_score(self.Y_test,prediction)
        conf_matrix=confusion_matrix(self.Y_test,prediction)
        print('accuracy %f'%acc_score)
        print('confusion matrix \n',conf_matrix)

        #do ROC for each k
        # plot misclassification error vs k
        plt.plot(self.neighbors, MSE)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Misclassification Error')
        plt.savefig('MSEvsk.png')
        #print(model.score)

    @staticmethod
    def plot_loss(losses, title=None):
        fig = pyplot.gcf()
        gene=title.split(' ')[0]
        fig.set_size_inches(8,6)
        ax = pyplot.axes()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        x_loss = list(range(len(losses)))
        pyplot.plot(x_loss, losses, color='red', label='Loss')
        pyplot.legend(loc='upper right')
        pyplot.title(title)
        if gene=='Disease':
            pyplot.savefig('./DiseasePredictionLoss.png',dpi=300)
        elif gene=='Mutation':
            pyplot.savefig('./MutationPredictionLoss.png',dpi=300)
        else:
            pyplot.savefig('./%s_Loss.png'%gene,dpi=300)

        pyplot.close()

    @staticmethod
    def plot_accuracy(accuracies, title=None):
        fig = pyplot.gcf()
        gene=title.split(' ')[0]
        fig.set_size_inches(8,6)
        ax = pyplot.axes()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("accuracy")
        x_loss = list(range(len(accuracies)))
        pyplot.plot(x_loss, accuracies, color='blue', label='Accuracy')
        pyplot.legend(loc='upper right')
        pyplot.title(title)
        if gene=='Disease':
            pyplot.savefig('./DiseasePredictionAccuracy.png',dpi=300)
        elif gene=='Mutation':
            pyplot.savefig('./MutationPredictionLoss.png',dpi=300)
        else:
            pyplot.savefig('./%s_Accuracy.png'%gene,dpi=300)

        pyplot.close()

    def ROC(self):
        pass


X,Y=preprocessing(iris)
X_train, X_test,Y_train,Y_test=split_data(X,Y)
classify=Classifier(X_train, X_test,Y_train,Y_test,kfolds=10, neighbors=10)
print('logistic regression results:\n')
classify.lr()
print('KNN results:')
classify.knn()
