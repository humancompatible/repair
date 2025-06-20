import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from metrics import DisparateImpact_postprocess


class ROCpostprocess:
    def __init__(self,X_val,y_val,var_list,prediction_model,favorable_label):
        self.X_val = X_val
        self.y_val = y_val
        self.model = prediction_model
        self.positive_index = 1 # positive label
        self.var_list = var_list
        self.var_dim=len(self.var_list)
        self.ROC = self.buildROCusingval()
        self.favorable_label = favorable_label

    def buildbinarydata(self,X,y):
        df=pd.DataFrame(np.concatenate((X,y.reshape(-1,1)), axis=1),columns=self.var_list+['S','W','Y'])
        binaryLabelDataset = BinaryLabelDataset(
                            # favorable_label=self.favorable_label,
                            # unfavorable_label=0,
                            df=df[self.var_list+['S','W','Y']], #df_test.drop('X',axis=1), #[x_list+['S','W','Y']],
                            label_names=['Y'],
                            instance_weights_name=['W'],
                            protected_attribute_names=['S'],
                            privileged_protected_attributes=[np.array([1.0])],
                            unprivileged_protected_attributes=[np.array([0.])])
        return binaryLabelDataset,df

    def buildROCusingval(self):
        dataset_val = self.buildbinarydata(self.X_val,self.y_val)[0]
        dataset_val_pred = dataset_val.copy(deepcopy=True)
        dataset_val_pred.scores = self.model.predict_proba(dataset_val.features[:,0:self.var_dim])[:,self.positive_index].reshape(-1,1)
        privileged_groups = [{'S': 1}]
        unprivileged_groups = [{'S': 0}]
        # Metric used (should be one of allowed_metrics)
        metric_name = "Statistical parity difference"
        # Upper and lower bound on the fairness metric used
        metric_ub = 0.05
        metric_lb = -0.05
        ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                        privileged_groups=privileged_groups, 
                                        low_class_thresh=0.01, high_class_thresh=0.99,
                                        num_class_thresh=50, num_ROC_margin=10,
                                        metric_name=metric_name,
                                        metric_ub=metric_ub, metric_lb=metric_lb)
        ROC = ROC.fit(dataset_val, dataset_val_pred)
        print("Optimal classification threshold (with fairness constraints) = %.4f" % ROC.classification_threshold)
        print("Optimal ROC margin = %.4f" % ROC.ROC_margin)
        return ROC

    def postprocess(self,X_test,y_test,tv_origin): # the tv distance won't change
        dataset_test_pred,df_test = self.buildbinarydata(X_test,y_test) #.copy(deepcopy=True)
        dataset_test_pred.scores = self.model.predict_proba(X_test[:,0:self.var_dim])[:,self.positive_index].reshape(-1,1)
        dataset_test_pred_transf = self.ROC.predict(dataset_test_pred)
        y_pred = dataset_test_pred_transf.labels
        # return dataset_test_pred_transf.convert_to_dataframe()[0]

        di = DisparateImpact_postprocess(df_test,y_pred,favorable_label=self.favorable_label)
        f1_macro = f1_score(df_test['Y'], y_pred, average='macro',sample_weight=df_test['W'])
        f1_micro = f1_score(df_test['Y'], y_pred, average='micro',sample_weight=df_test['W'])
        f1_weighted = f1_score(df_test['Y'], y_pred, average='weighted',sample_weight=df_test['W'])
        new_row=pd.Series({'DI':di,'f1 macro':f1_macro,'f1 micro':f1_micro,'f1 weighted':f1_weighted,
                           'TV distance':tv_origin,'method':'ROC'})
        return new_row.to_frame().T
    
