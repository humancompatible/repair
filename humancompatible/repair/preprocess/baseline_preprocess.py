import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from aif360.algorithms.preprocessing import DisparateImpactRemover,Reweighing,LFR


class Baselinepreprocess:
    """
    A class to evaluate fairness and performance of 3 bias mitigation methods
    in the AIF360 documentation:https://aif360.readthedocs.io/en/latest/modules/algorithms.html.

    This class supports methods like Reweighing, Disparate Impact Remover, and LFR (Learning fair representations)
    to preprocess data, train models, and assess fairness metrics of 
    Disparate Impact and F1 scores.

    Parameters:
        train, test (CompasDataset): The dataset to be evaluated.
        pa (str): The name of the protected attribute (e.g., 'sex', 'race').
   
    Methods:
        preprocessing(method): Preprocess the dataset using the specified method.
        prediction(method): Predict outcomes using a random forest on the test data.
        assess(method): Compute performance and fairness metrics.
    """

    def __init__(self,train,test):
        self.train = train 
        self.test = test
        self.pa = train.protected_attribute_names[0]
        self.pa_index = train.feature_names.index(self.pa)
        self.prigroups = [{self.pa: 1}]
        self.unprigroups = [{self.pa: 0}]

    def preprocessing(self,method):
        """
        Preprocess training and/or test data for a given fairness method.

        Applies preprocessing steps as described in the AIF360 documentation:
        https://aif360.readthedocs.io/en/latest/modules/algorithms.html

        Parameters:
            methods (str): The name of the method to evaluate.
                        Must be one of ['origin', 'RW', 'DIremover', 'LFR'].

        Returns:
            CompasDataset: The processed training and test data.
        """
        test_tranf = self.test.copy()
        if method == 'RW':
            RW = Reweighing(privileged_groups = self.prigroups,
                            unprivileged_groups = self.unprigroups)
            RW.fit(self.train)
            train_tranf = RW.transform(self.train)
        elif method == 'DIremover':
            di = DisparateImpactRemover(repair_level = 1,
                                        sensitive_attribute=self.pa)
            train_tranf = di.fit_transform(self.train)
            test_tranf = di.fit_transform(self.test)
        elif method == 'LFR':
            TR = LFR(privileged_groups = self.prigroups,
                     unprivileged_groups = self.unprigroups,
                     Az = 1, Ax = 0.01, Ay = 1,verbose=0)
            TR = TR.fit(self.train)
            train_tranf = TR.transform(self.train)
            test_tranf = TR.transform(self.test)
        return train_tranf, test_tranf

    def prediction(self,method):
        """
        Predict outcomes using a random forest classifier with a given fairness method.

        Parameters:
            methods (str): The name of the method to evaluate.
                        Must be one of ['origin', 'RW', 'DIremover', 'LFR'].

        Returns:
            y_pred (CompasDataset): Predictions on the test data.
            di (float): Disparate Impact computed on the (processed) training data.
        """
        test_tranf = self.test.copy()
        if method == 'origin':
            train_tranf = self.train
        elif method in ['RW','DIremover','LFR','OP']:
            train_tranf,test_tranf = self.preprocessing(method)
        else:
            print('The method does not exist')

        di=self.DisparateImpact(train_tranf)
        print('Disparate Impact of train',di)

        if method != 'LFR':
            X_train = np.delete(train_tranf.features, self.pa_index, axis=1)
            y_train = train_tranf.labels.ravel()
            weight_train = train_tranf.instance_weights
            model=RandomForestClassifier(max_depth=5).fit(X_train,y_train, sample_weight=weight_train)

            X_test = np.delete(test_tranf.features, self.pa_index, axis=1)
            y_pred = model.predict(X_test)
        else:
            y_pred = test_tranf.labels
        return y_pred,di
    
    def DisparateImpact(self,data):
        """
        Computes Disparate Impact of the given dataset.

        Parameters:
            data (CompasDataset).
        """
        di = pd.DataFrame({'S':data.protected_attributes.ravel().tolist(),
            'Y':data.labels.ravel().tolist(),
            'W':list(data.instance_weights)},columns=['S','Y','W'])
        privileged = self.train.privileged_protected_attributes[0][0]
        unprivileged = self.train.unprivileged_protected_attributes[0][0]
        numerator=sum(di[(di['S']==unprivileged)&(di['Y']==data.favorable_label)]['W'])/sum(di[di['S']==unprivileged]['W'])
        denominator=sum(di[(di['S']==privileged)&(di['Y']==data.favorable_label)]['W'])/sum(di[di['S']==privileged]['W'])
        if numerator==denominator:
            return 1
        return numerator/denominator

    def assess(self,method):
        """
        Calculate performance metrics for a given fairness method.

        Computes Disparate Impact and three types of F1 scores of the prediction on (processed) test data.

        Parameters:
            methods (str): The name of the method to evaluate.
                        Must be one of ['origin', 'RW', 'DIremover', 'LFR'].

        Returns:
            pd.DataFrame: A DataFrame containing the performance metrics
                        for the specified method.
        """
        y_pred,di_train = self.prediction(method)
        y_test_pred = self.test.copy()
        y_test_pred.labels = y_pred

        di=self.DisparateImpact(y_test_pred)
        f1_macro = f1_score(self.test.labels, y_pred, average='macro',sample_weight=self.test.instance_weights)
        f1_micro = f1_score(self.test.labels, y_pred, average='micro',sample_weight=self.test.instance_weights)
        f1_weighted = f1_score(self.test.labels, y_pred, average='weighted',sample_weight=self.test.instance_weights)
        print('Disparate Impact of '+str(method),di)
        print('f1 macro of '+str(method),f1_macro)

        new_row=pd.Series({'DI of train':di_train,'DI':di,'f1 macro':f1_macro,'f1 micro':f1_micro,'f1 weighted':f1_weighted,'method':method})
        return new_row.to_frame().T
