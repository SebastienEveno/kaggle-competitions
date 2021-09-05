import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

class Model:
    def __init__(self, estimator: BaseEstimator, grid: dict) -> None:
        self.estimator = estimator
        self.grid = grid

class ClassificationModels:
    def __init__(self) -> None:
        self.models = [
                Model(LogisticRegression(max_iter=10000), {'solver': ['lbfgs', 'liblinear'], 'C': [0.01, 0.1, 1, 10], 'penalty': ["l1","l2"]}),
                Model(LinearDiscriminantAnalysis(), {'solver': ['svd', 'lsqr', 'eigen']}),
                Model(QuadraticDiscriminantAnalysis(), {'reg_param': [0.001, 0.01, 0.05, 0.1, 0.5]}),
                # Model(SGDClassifier(max_iter=1000, tol=1e-3)),
                # Model(LinearSVC(C=1.0, max_iter=1000, tol=1e-3, dual=False)),
                # Model(RadiusNeighborsClassifier(radius=40.0)),
                # Model(KNeighborsClassifier()),
                # Model(DecisionTreeClassifier()),
                # Model(GaussianNB())
            ]
    
class ClassificationManager:
    def __init__(self, features: pd.DataFrame, target_variable: pd.DataFrame) -> None:
        self.X = features
        self.y = target_variable

    def build_all_models(self) -> dict:
        models = ClassificationModels().models
        results = dict()
        for model in models:
            results[model.estimator.__class__.__name__] = self.build_model(model)
        
        self.compare_results(results)
        return results
    
    def build_model(self, model: Model, test_frac=0.2) -> dict:

        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_frac)
        
        # Perform GridSearch
        estimator = GridSearchCV(model.estimator, model.grid, cv=10)
        estimator = estimator.fit(x_train, y_train)

        y_pred = estimator.predict(x_test)
        y_pred_train = estimator.predict(x_train)

        train_summary = self.summarize(y_train, y_pred_train)
        test_summary = self.summarize(y_test, y_pred)

        pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

        return { 'training': train_summary, 'test': test_summary, 
        'confusion_matrix': model_crosstab, 'best_params': estimator.best_params_,
        'best_score': estimator.best_score_ }

    def summarize(self, y_test: list, y_pred: list) -> dict:

        acc = accuracy_score(y_test, y_pred, normalize=True)
        num_acc = accuracy_score(y_test, y_pred, normalize=False)

        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        return { 'accuracy': acc, 'precision': prec, 
        'recall': recall, 'accuracy_count': num_acc }

    def compare_results(self, results: dict) -> None:
        for key in results:
            print('Classification model: ', key)

            print()
            print('Training data')
            for score in results[key]['training']:
                print(score, results[key]['training'][score])
            
            print()
            print('Test data')
            for score in results[key]['test']:
                print(score, results[key]['test'][score])
            
            print()
            print('Best score: ', results[key]['best_score'])
            print('Best params: ', results[key]['best_params'])

            print()