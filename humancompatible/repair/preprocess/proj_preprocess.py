import numpy as np

from aif360.datasets import BinaryLabelDataset

from methods.cost import c_generate, c_generate_higher
from methods.data_analysis import rdata_analysis
from group_blind_repair import GroupBlindRepair
from methods.coupling_utils import projection, projection_higher


class Projpreprocess:
    
    def __init__(self,traindata,x_list,var_list,K,e):
        self.K=K
        self.e=e
        self.x_list=x_list
        self.var_list=var_list
      
        self.var_dim=len(var_list)
        self.arg_list=[elem for elem in var_list if elem not in x_list]
        self.train = traindata.copy()
        self.df = self.train.convert_to_dataframe()[0]
        self.pa = self.train.protected_attribute_names[0]
        self.pa_index = self.train.feature_names.index(pa)
        self.label_name = self.train.label_names[0]
        self.df=self.df.rename(columns={self.pa:'S',self.label_name:'Y'})

        self.df['W'] = self.train.instance_weights
        for col in self.var_list+['S','Y']:
            self.df[col]=self.df[col].astype('int64')
        self.df=self.df[var_list+['S','W','Y']]
        if len(x_list)>1:
            self.df['X'] = list(zip(*[self.df[c] for c in x_list]))
            self.x_range=sorted(set(self.df['X']))
            weight=list(1/(self.df[x_list].max()-self.df[x_list].min())) # because ranges of attributes differ
            self.C=c_generate_higher(self.x_range,weight)
        else:
            self.df['X']=self.df[x_list]
            self.x_range=sorted(set(self.df['X']))
            self.C=c_generate(self.x_range)
        self.df = self.df[self.arg_list+['X','S','Y','W']].groupby(by=self.arg_list+['X','S','Y'],as_index=False).sum()
        self.distribution_generator()
        
    def distribution_generator(self):
        bin=len(self.x_range)
        dist=rdata_analysis(self.df,self.x_range,'X')
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
        # self.tv_origin=sum(abs(dist['x_0']-dist['x_1']))/2
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

    def coupling_generator(self,method,Theta=1e-2):
        if method == 'unconstrained':
            coupling=self._run_method(method="baseline", C=self.C, eps=self.e, px=self.px, ptx=self.ptx, K=self.K)
        elif method == 'barycentre':
            coupling=self._run_method(method="baseline", C=self.C, eps=self.e, px=self.p0, ptx=self.p1, K=self.K)
        elif method == 'partial':
            coupling=self._run_method(method="partial_repair", C=self.C, eps=self.e, px=self.px, ptx=self.ptx, V=self.V, theta=Theta, K=self.K)
        return coupling

    def preprocess(self,method,Theta=1e-2):
        coupling = self.coupling_generator(method,Theta)
        if len(self.x_list)>1:
            df_proj=projection_higher(self.df,coupling,self.x_range,self.x_list,self.var_list)
        else:
            df_proj=projection(self.df,coupling,self.x_range,self.x_list[0],self.var_list)
        df_proj = df_proj.groupby(by=self.arg_list+['X','S','Y'],as_index=False).sum()
        X=list(zip(*df_proj['X']))
        df_proj = df_proj.assign(**{self.x_list[i]:X[i] for i in range(len(self.x_list))})
        df_proj=df_proj.drop('X',axis=1)
        df_proj=df_proj.rename(columns={'S':self.pa,'Y':self.label_name})
        binaryLabelDataset = BinaryLabelDataset(
            favorable_label=0,
            unfavorable_label=1,
            df=df_proj.drop('W',axis=1), 
            label_names=self.train.label_names,
            protected_attribute_names=self.train.protected_attribute_names,
            privileged_protected_attributes=[np.array([1.0])],unprivileged_protected_attributes=[np.array([0.])])
        binaryLabelDataset.instance_weights = df_proj['W'].tolist()
        # return binaryLabelDataset.align_datasets(self.train)
        return self.train.align_datasets(binaryLabelDataset)
    
