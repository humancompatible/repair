import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from cost import c_generate, c_generate_higher
from data_analysis import rdata_analysis
from group_blind_repair import GroupBlindRepair
from coupling_utils import postprocess, postprocess_bary
from metrics import assess_tv, DisparateImpact_postprocess


class Projpostprocess:
    
    def __init__(self,X_test,y_test,x_list,var_list,prediction_model,K,e,thresh,favorable_label=1,**kwargs):
        self.model = prediction_model
        self.K=K
        self.e=e
        self.x_list=x_list
        self.var_list=var_list
        self.var_dim=len(var_list)
        self.favorable_label = favorable_label

        self.linspace_range = kwargs.get('linspace_range', (0.01, 0.1))
        self.theta = kwargs.get('theta', 1e-2)
        print("EXTRA PARAMS ===========================")
        print(self.linspace_range)
        print(self.theta)

        df_test=pd.DataFrame(np.concatenate((X_test,y_test.reshape(-1,1)), axis=1),columns=var_list+['S','W','Y'])
        df_test=df_test.groupby(by=var_list+['S','Y'],as_index=False).sum()
        if len(x_list)>1:
            df_test['X'] = list(zip(*[df_test[c] for c in x_list]))
            self.x_range=sorted(set(df_test['X']))
            weight=list(1/(df_test[x_list].max()-df_test[x_list].min())) # because 'education-num' range from 1 to 16 while others 1 to 4
            self.C=c_generate_higher(self.x_range,weight)
        else:
            df_test['X']=df_test[x_list]
            self.x_range=sorted(set(df_test['X']))
            self.C=c_generate(self.x_range)
        self.df_test = df_test
        self.var_range=list(pd.pivot_table(df_test,index=var_list,values=['S','W','Y']).index)
        self.distribution_generator()
        
        if thresh == 'auto':
            self.thresh_generator()
        else:
            self.thresh=thresh

    def distribution_generator(self):
        bin=len(self.x_range)
        dist=rdata_analysis(self.df_test,self.x_range,'X')
        
        dist['v']=[(dist['x_0'][i]-dist['x_1'][i])/dist['x'][i] for i in range(bin)]
        
        dist['t_x']=dist['x'] # #dist['x'] #dist['x_0']*0.5+dist['x_1']*0.5 
        self.px=np.matrix(dist['x']).T
        self.ptx=np.matrix(dist['t_x']).T
        if np.any(dist['x_0']==0): 
            self.p0=np.matrix((dist['x_0']+1.0e-9)/sum(dist['x_0']+1.0e-9)).T
        else:
            self.p0=np.matrix(dist['x_0']).T 
        if np.any(dist['x_1']==0):
            self.p1=np.matrix((dist['x_1']+1.0e-9)/sum(dist['x_1']+1.0e-9)).T
        else:
            self.p1=np.matrix(dist['x_1']).T 
        self.V=np.matrix(dist['v']).T
        self.tv_origin=sum(abs(dist['x_0']-dist['x_1']))/2
        # return px,ptx,V,p0,p1
    
    def _run_method(self, method, C, eps, px, ptx, K, V=None, theta=None):
        group_blind = GroupBlindRepair(C, px, ptx, V=V, epsilon=eps, K=K)
        if method == "baseline":
            group_blind.fit_baseline()
        elif method == "partial_repair":
            group_blind.fit_partial(theta)
        elif method == "total_repair":
            group_blind.fit_total()
        return group_blind.coupling_matrix()

    def coupling_generator(self,method,para=None):
        if method == 'unconstrained':
            coupling=self._run_method(method="baseline", C=self.C, eps=self.e, px=self.px, ptx=self.ptx, K=self.K)
        elif method == 'barycentre':
            coupling=self._run_method(method="baseline", C=self.C, eps=self.e, px=self.p0, ptx=self.p1, K=self.K)
        elif method == 'partial':
            coupling=self._run_method(method="partial_repair", C=self.C, eps=self.e, px=self.px, ptx=self.ptx, V=self.V, theta=para, K=self.K)
        return coupling

    def postprocess(self,method,para=None):
        if method == 'origin':
            y_pred=self.model.predict(np.array(self.df_test[self.var_list]))
            tv = self.tv_origin
        else:
            coupling = self.coupling_generator(method,para)
            if (method == 'unconstrained') or (method == 'partial'):
                y_pred=postprocess(self.df_test,coupling,self.x_list,self.x_range,self.var_list,self.var_range,self.model,self.thresh)
                tv=assess_tv(self.df_test,coupling,self.x_range,self.x_list,self.var_list)
                if (para != None) and (method == 'partial'):
                    method = method+'_'+str(para)
            elif method == 'barycentre':
                y_pred,tv=postprocess_bary(self.df_test,coupling,self.x_list,self.x_range,self.var_list,self.var_range,self.model,self.thresh)
            else:
                print('Unknown method')

        di = DisparateImpact_postprocess(self.df_test,y_pred,favorable_label=self.favorable_label)
        f1_macro = f1_score(self.df_test['Y'], y_pred, average='macro',sample_weight=self.df_test['W'])
        f1_micro = f1_score(self.df_test['Y'], y_pred, average='micro',sample_weight=self.df_test['W'])
        f1_weighted = f1_score(self.df_test['Y'], y_pred, average='weighted',sample_weight=self.df_test['W'])

        new_row=pd.Series({'DI':di,'f1 macro':f1_macro,'f1 micro':f1_micro,'f1 weighted':f1_weighted,
                           'TV distance':tv,'method':method})
        return new_row.to_frame().T
    
    def thresh_generator(self):
        num_thresh = 10
        ba_arr = np.zeros(num_thresh)
        ba_arr1 = np.zeros(num_thresh)

        ## For notebooks
        # "adult" and "compas":
        #   class_thresh_arr = np.linspace(0.01, 0.1, num_thresh)
        #   coupling=self.coupling_generator('partial',para=1e-2)
        #
        # "adult_new":
        #   class_thresh_arr = np.linspace(0.1, 0.9, num_thresh)
        #   coupling=self.coupling_generator('partial',para=1e-3)
        ##
        class_thresh_arr = np.linspace(self.linspace_range[0], self.linspace_range[1], num_thresh)
        coupling=self.coupling_generator('partial',para=self.theta)
    
        for idx, thresh in enumerate(class_thresh_arr):
            y_pred=postprocess(self.df_test,coupling,self.x_list,self.x_range,self.var_list,self.var_range,self.model,thresh)
            ba_arr[idx] = DisparateImpact_postprocess(self.df_test,y_pred,favorable_label=self.favorable_label)
            ba_arr1[idx] = f1_score(self.df_test['Y'], y_pred, average='macro',sample_weight=self.df_test['W'])

        best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
        best_thresh = class_thresh_arr[best_ind]
        print("Optional threshold = ",class_thresh_arr)
        print("Disparate Impact = ",ba_arr)
        print("f1 scores = ",ba_arr1)
        self.thresh = best_thresh

