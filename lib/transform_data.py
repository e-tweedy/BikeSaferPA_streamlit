import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, SplineTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class GroupImputer(BaseEstimator, TransformerMixin):
    """
    Class used for imputing missing values in a pd.DataFrame
    using mean, median, or mode by groupwise aggregation,
    or a constant.
    
    Parameters:
    -----------
    target : str 
        - The name of the column to be imputed
    group_cols : list
        - List of name(s) of columns on which to groupby
    strategy : str
         - The method for replacement; can be any of
          ['mean', 'median', 'mode']
    
    Returns:
    --------
    X : pd.DataFrame
        - The dataframe with imputed values in the target column
    
    """
    def __init__(self,target,group_cols=None,strategy='median'):
        assert strategy in ['mean','median','mode'], "strategy must be in ['mean', 'median', 'mode']'"
        assert type(group_cols)==list, 'group_cols must be a list of column names'
        assert type(target) == str, 'target must be a string'
        
        self.group_cols = group_cols
        self.strategy=strategy
        self.target = target
        
    def fit(self,X,y=None):
        
        if self.strategy=='mode':
            impute_map = X.groupby(self.group_cols)[self.target]\
                            .agg(lambda x: pd.Series.mode(x,dropna=False)[0])\
                            .reset_index(drop=False)
        else:
            impute_map = X.groupby(self.group_cols)[self.target]\
                        .agg(self.strategy).reset_index(drop=False)
        self.impute_map_ = impute_map
        
        return self
        
    def transform(self,X,y=None):
        
        check_is_fitted(self,'impute_map_')
        
        X=X.copy()
        
        for index,row in self.impute_map_.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind,self.target] = X.loc[ind,self.target].fillna(row[self.target])
        return X
	
# Sine and consine transformations
def sin_feature_names(transformer, feature_names):
    return [f'SIN_{col}' for col in feature_names]
def cos_feature_names(transformer, feature_names):
    return [f'COS_{col}' for col in feature_names]    
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(2*np.pi*x/period),feature_names_out = sin_feature_names)
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(2*np.pi*x/period),feature_names_out = cos_feature_names)

# Periodic spline transformation
def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )