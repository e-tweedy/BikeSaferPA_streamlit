import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, f1_score, fbeta_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer, SplineTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from lib.transform_data import *

class ClassifierStudy():
    """
    A class that contains tools for studying a classifier pipeline
    
    Parameters:
    -----------
    classifier : a scikit-learn compatible binary classifier
    X : pd.DataFrame
        dataframe of features
    y : pd.Series
        series of binary target values corresponding to X
    classifier_name : str or None
        if provided, will use as classifier name in pipeline
        if not, will use 'clf' as name
    features : dict
        a dictionary whose keys are the feature types
        'cyc','cat','ord','num','bin' and whose values
        are lists of features of each type.
        
    Methods:
    -------
    set_data, set_features, set_state
        sets or resets attributes of self
    build_pipeline
        builds out pipeline based on supplied specs
    cv_score
        runs k-fold cross validation and reports scores
    randomized_search
        runs randomized search with cross validation
        and reports results
    fit_pipeline
        fits the model pipeline and stores as
        self.pipe_fitted
    predict_proba_pipeline
        uses a fitted pipeline to compute predicted
        probabilities for test or validation set
    score_pipeline
        scores predicted probabilities
        
    """
    def __init__(self, classifier=None, X = None, y = None,
                 features = None,classifier_name = None,
                 random_state=42):
        self.classifier = classifier
        if X is not None:
            self.X = X.copy()
        if y is not None:
            self.y = y.copy()
        if features is not None:
            self.features = features.copy()
        self.random_state=random_state
        self.pipe, self.pipe_fitted = None, None
        self.classifier_name = classifier_name
        self.X_val, self.y_val = None, None
        self.y_predict_proba = None
        self.best_params, self.best_n_components = None, None
        self.shap_vals = None
    
    def set_data(self,X=None,y=None):
        """Method to set or reset feature and/or target data"""
        if X is not None:
            self.X = X.copy()
        if y is not None:
            self.y = y.copy()
    
    def set_features(self,features):
        """Method to set or reset the feature dictionary"""
        if features is not None:
            self.features = features.copy()        
    
    def set_state(self,random_state):
        """Method to set or reset the random_state"""
        self.random_state = random_state
        
    def build_pipeline(self, cat_method = 'onehot',cyc_method = 'spline',num_ss=True,
                       over_sample = False, pca=False,n_components=None,
                       select_features = False,score_func=None,k='all',
                       poly_features = False, degree=2, interaction_only=False):
        """
        Method to build the model pipeline
        Parameters:
        -----------
        cat_method : str
            specifies whether to encode categorical
            variables as one-hot vectors or ordinals
            must be either 'onehot' or 'ord'
        cyc_method : str
            specifies whether to encode cyclical features
            with sine/cosine encoding or periodic splines
            must be one of 'trig', 'spline', 'interact-trig',
            'interact-spline','onehot', 'ord', or None
            - If 'trig' or 'spline', will set up periodic encoder
              with desired method
            - If 'onehot' or 'ord', will set up appropriate
              categorical encoder
            - If 'interact-{method}', will use <method> encoding for HOUR_OF_DAY,
              encode DAY_OF_WEEK as a binary feature expressing whether
              the day is a weekend day, and then include interaction
              features among this set via PolynomialFeatures.
            - If None, will leave out cyclical features altogether
        num_ss : bool
            Whether or not to apply StandardScaler on the numerical features
        over_sample : bool
            set to True to include imblearn.over_sampling.RandomOverSampler step
        pca : bool
            set to True to include sklearn.decomposition.PCA step
        n_components : int or None
            number of components for sklearn.decomposition.PCA
        select_features : bool
            set to True to include sklearn.feature_selection.SelectKBest step
        score_func : callable
            score function to use for sklearn.feature_selection.SelectKBest
            recommended: chi2, f_classif, or mutual_info_classif
        k : int or 'all'
            number of features for sklearn.feature_selection.SelectKBest
        poly_features : bool
            set to True to include sklearn.preprocessing.PolynomialFeatures step
        degree : int
            max degree for sklearn.preprocessing.PolynomialFeatures
        interaction_only : bool
            whether or not sklearn.preprocessing.PolynomialFeatures will be limited
            to interaction terms only
        """
        
        # Define transformer for categorical features
        if cat_method == 'onehot':
            cat_encoder = ('ohe',OneHotEncoder(handle_unknown='infrequent_if_exist'))
                                
        elif cat_method == 'ord':
            cat_encoder = ('oe',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan))
        else:
            raise ValueError("cat_method must be either 'onehot' or 'ord'")
    
        cat_transform = Pipeline([('si',SimpleImputer(strategy='most_frequent')),cat_encoder])
    
        # Define transformer for cyclic features
        cyc_dict = {'HOUR_OF_DAY':24,'DAY_OF_WEEK':7}
        if cyc_method == 'trig':
            cyc_transform = [(f'{feat}_cos',cos_transformer(cyc_dict[feat]),[feat]) for feat in self.features['cyc']]+\
                        [(f'{feat}_sin',sin_transformer(cyc_dict[feat]),[feat]) for feat in self.features['cyc']]
        elif cyc_method =='spline':
            cyc_transform = [(f'{feat}_cyclic',
                          periodic_spline_transformer(cyc_dict[feat],n_splines=cyc_dict[feat]//2),
                          [feat]) for feat in self.features['cyc']]
        elif cyc_method == 'onehot':
            cyc_encoder = ('ohe_cyc',OneHotEncoder(handle_unknown='infrequent_if_exist'))
            cyc_transform = [('cyc',Pipeline([cyc_encoder]),self.features['cyc'])]
        elif cyc_method == 'ord':
            cyc_encoder = ('oe_cyc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan))
            cyc_transform = [('cyc',Pipeline([cyc_encoder]),self.features['cyc'])]
        elif cyc_method == 'interact-spline':
            hour_transform = (f'hour_cyc',periodic_spline_transformer(cyc_dict['HOUR_OF_DAY'],n_splines=12),['HOUR_OF_DAY'])
            wkend_transform = ('wkend',FunctionTransformer(lambda x: (x.isin([1,7])).astype(int)),['DAY_OF_WEEK'])
            cyc_transform = [('cyc',Pipeline([('cyc_col',ColumnTransformer([hour_transform, wkend_transform],
                                                               remainder='drop',verbose_feature_names_out=False)),
                                              ('cyc_poly',PolynomialFeatures(degree=2,interaction_only=True,
                                                                include_bias=False))]),
                             self.features['cyc'])]
        elif cyc_method == 'interact-trig':
            hour_transform = [(f'HOUR_cos',cos_transformer(cyc_dict['HOUR_OF_DAY']),['HOUR_OF_DAY']),
                              (f'HOUR_sin',sin_transformer(cyc_dict['HOUR_OF_DAY']),['HOUR_OF_DAY'])]
            wkend_transform = ('wkend',FunctionTransformer(lambda x: (x.isin([1,7])).astype(int)),['DAY_OF_WEEK'])
            cyc_transform = [('cyc',Pipeline([('cyc_col',ColumnTransformer(hour_transform+[wkend_transform],
                                                               remainder='drop',verbose_feature_names_out=False)),
                                              ('cyc_poly',PolynomialFeatures(degree=2,interaction_only=True,
                                                                include_bias=False))]),
                             self.features['cyc'])]
        elif cyc_method is None:
            cyc_transform = [('cyc','passthrough',[])]
        else:
            raise ValueError("cyc_method must be one of 'trig','spline','interact','onehot','ord',or None")
        
        # Define numerical transform
        num_transform = ('num',StandardScaler(),self.features['num']) if num_ss else\
                        ('num','passthrough',self.features['num'])
        
        # Define column transformer
        col_transform = ColumnTransformer([('cat',cat_transform,self.features['cat']),
                                           ('ord','passthrough',self.features['ord']),
                                           num_transform,
                                           ('bin',SimpleImputer(strategy='most_frequent'),
                                             self.features['bin'])]+\
                                           cyc_transform,
                                           remainder='drop',verbose_feature_names_out=False)
    
        steps = [('col',col_transform)]
    
        if 'AGE' in self.features['num']:
            steps.insert(0,('gi_age',GroupImputer(target = 'AGE', group_cols=['COUNTY'],strategy='median')))
        if 'HOUR_OF_DAY' in self.features['cyc']:
            steps.insert(0,('gi_hour',GroupImputer(target = 'HOUR_OF_DAY', group_cols=['ILLUMINATION','CRASH_MONTH'],strategy='mode')))
        # Insert optional steps as needed
        if over_sample:
            steps.insert(0,('os',RandomOverSampler(random_state=self.random_state)))
        if poly_features:
            steps.append(('pf',PolynomialFeatures(degree=degree,interaction_only=interaction_only)))
        if select_features:
            steps.append(('fs',SelectKBest(score_func = score_func, k = k)))
        if pca:
            steps.append(('pca',PCA(n_components=n_components,random_state=self.random_state)))
        # Append classifier if provided
        if self.classifier is not None:
            if self.classifier_name is not None:
                steps.append((f'{self.classifier_name}_clf',self.classifier))
            else:
                steps.append(('clf',self.classifier))
    
        # Initialize pipeline
        self.pipe = Pipeline(steps)
    
    def cv_score(self, scoring = 'roc_auc', n_splits = 5, n_repeats=3, thresh = 0.5, beta = 1,
                 return_mean_score=False,print_mean_score=True,print_scores=False, n_jobs=-1,
                eval_size=0.1,eval_metric='auc'):
        """
        Method for performing cross validation via RepeatedStratifiedKFold
        
        Parameters:
        -----------
        scoring : str
            scoring function to use.  must be one of
            'roc_auc','acc','f1','','f1w'
        thresh : float
            the classification threshold for computing y_pred
            from y_pred_proba
        beta : float
            the beta-value to use in the f_beta score, if chosen
        n_splits, n_repeats : int, int
            number of splits and number of repeat iterations
            for sklearn.model_selection.RepeatedStratifiedKFold
        return_mean_score : bool
            whether or not to return the mean score
        print_mean_score : bool
            whether to print out a report of the mean score
        print_scores : bool
            whether to print out a report of CV scores for all folds
        n_jobs : int or None
            number of CPU cores to use for parallel processing
            -1 uses all available cores, and None defaults to 1
        eval_size : float
            Fraction of the training set to use for early stopping eval set
        eval_metric : str
            eval metric to use in early stopping
        Returns: None or mean_score, depending on return_mean_score setting
        --------
        """
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert (self.X is not None)&(self.y is not None), 'X and/or y does not exist.  First supply X and y using set_data.'
        assert 'clf' in self.pipe.steps[-1][0], 'The pipeline has no classifier.  Build a pipeline with a classifier first.'
        assert scoring in ['roc_auc','acc','f1','fb','f1w'],"scoring must be one of 'roc_auc','acc','f1','fb','f1w'"
        
        # Initialize CV iterator
        kf = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats,
                                     random_state=self.random_state)
        # Restrict to features supplied in self.features
        X = self.X[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        
        xgb_es=False
        if isinstance(self.pipe[-1],XGBClassifier):
            if self.pipe[-1].get_params()['early_stopping_rounds'] is not None:
                xgb_es=True

        scores = []
        # Iterate over folds and train, predict, score
        for i,(train_idx,test_idx) in enumerate(kf.split(X,self.y)):
            fold_X_train = X.iloc[train_idx,:]
            fold_X_test = X.iloc[test_idx,:]
            fold_y_train = self.y.iloc[train_idx]
            fold_y_test = self.y.iloc[test_idx]
            
            pipe=clone(self.pipe)
            if xgb_es:
                fold_X_train,fold_X_es,fold_y_train,fold_y_es = train_test_split(fold_X_train,fold_y_train,
                                                                                 stratify=fold_y_train,test_size=eval_size,
                                                                                 random_state=self.random_state)
                trans_pipe = pipe[:-1]
                trans_pipe.fit_transform(fold_X_train)
                fold_X_es = trans_pipe.transform(fold_X_es)
                clf_name = pipe.steps[-1][0]
                fit_params = {f'{clf_name}__eval_set':[(fold_X_es,fold_y_es)],
                              f'{clf_name}__eval_metric':eval_metric,
                              f'{clf_name}__verbose':0}
            else:
                fit_params = {}
            
            pipe.fit(fold_X_train,fold_y_train,**fit_params)
            fold_y_pred_proba = pipe.predict_proba(fold_X_test)[:,1]
            
            if scoring == 'roc_auc':
                fold_score = roc_auc_score(fold_y_test, fold_y_pred_proba)
            else:
                fold_y_pred = (fold_y_pred_proba >= thresh).astype('int')
                if scoring == 'acc':
                    fold_score = accuracy_score(fold_y_test,fold_y_pred)
                elif scoring == 'f1':
                    fold_score = f1_score(fold_y_test,fold_y_pred)
                elif scoring == 'f1w':
                    fold_score = f1_score(fold_y_test,fold_y_pred,average='weighted')
                else:
                    fold_score = fbeta_score(fold_y_test,fold_y_pred,beta=beta)
            scores.append(fold_score)
        
        # Average and report
        mean_score = np.mean(scores)
        if print_scores:
            print(f'CV scores using {scoring} score: {scores} \nMean score: {mean_score}')
        if print_mean_score:
            print(f'Mean CV {scoring} score: {mean_score}')
        if return_mean_score:
            return mean_score
            
    def randomized_search(self, params, n_components = None, n_iter=10,
                          scoring='roc_auc',cv=5,refit=False,top_n=10, n_jobs=-1):
        """
        Method for performing randomized search with cross validation on a given dictionary of parameter distributions
        Also displays a table of results the best top_n iterations
        
        Parameters:
        ----------
        params : dict
            parameter distributions to use for RandomizedSearchCV
        n_components : int, or list, or None
            number of components for sklearn.decomposition.PCA
            - if int, will reset the PCA layer in self.pipe with provided value
            - if list, must be list of ints, which will be included in
              RandomizedSearchCV parameter distribution
        scoring : str
            scoring function for sklearn.model_selection.cross_val_score
        n_iter : int
            number of iterations to use in RandomizedSearchCV
        refit : bool
            whether to refit a final classifier with best parameters
            - if False, will only set self.best_params and self.best_score
            - if True, will set self.best_estimator in addition
        top_n : int or None
            if int, will display results from top_n best iterations only
            if None, will display all results
        n_jobs : int or None
            number of CPU cores to use for parallel processing
            -1 uses all available cores, and None defaults to 1
        """
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert (self.X is not None)&(self.y is not None), 'X and/or y does not exist.  First supply X and y using set_data.'
        assert 'clf' in self.pipe.steps[-1][0], 'The pipeline has no classifier.  Build a pipeline with a classifier first.'
        assert (n_components is None)|('pca' in self.pipe.named_steps), 'Your pipeline has no PCA step.  Build a pipeline with PCA first.'
        assert (len(params)>0)|(type(n_components)==list), 'Either pass a parameter distribution or a list of n_components values.'
        
        # Add estimator name prefix to hyperparams
        params = {self.pipe.steps[-1][0]+'__'+key:params[key] for key in params}
        
        # Process supplied n_components
        if type(n_components)==list:
            params['pca__n_components']=n_components
        elif type(n_components)==int:
            self.pipe['pca'].set_params(n_components=n_components)
        
        # Restrict to features supplied in self.features
        X = self.X[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        
        # Initialize rs and fit
        rs = RandomizedSearchCV(self.pipe, param_distributions = params,
                                n_iter=n_iter, scoring = scoring, cv = cv,refit=refit,
                                random_state=self.random_state, n_jobs=n_jobs)
        
        rs.fit(X,self.y)
    
        # Display top n scores
        results = rs.cv_results_
        results_df = pd.DataFrame(results['params'])
        param_names = list(results_df.columns)
        results_df[f'mean cv score ({scoring})']=pd.Series(results['mean_test_score'])
        results_df = results_df.set_index(param_names).sort_values(by=f'mean cv score ({scoring})',ascending=False)
        if top_n is not None:
            display(results_df.head(top_n).style\
                    .highlight_max(axis=0, props='color:white; font-weight:bold; background-color:seagreen;'))
        else:
            display(results_df.style\
                    .highlight_max(axis=0, props='color:white; font-weight:bold; background-color:seagreen;'))
        if refit:
            self.best_estimator = rs.best_estimator_
        best_params = rs.best_params_
        self.best_params = {key.split('__')[-1]:best_params[key] for key in best_params if key.split('__')[0]!='pca'}
        self.best_n_components = next((best_params[key] for key in best_params if key.split('__')[0]=='pca'), None)
        self.best_score = rs.best_score_
        
    def fit_pipeline(self,split_first=False, eval_size=0.1,eval_metric='auc'):
        """
        Method for fitting self.pipeline on self.X,self.y
        Parameters:
        -----------
        split_first : bool
            if True, a train_test_split will be performed first
            and the validation set will be stored
        early_stopping : bool
            Indicates whether we will use early_stopping for xgboost.
            If true, will split off an eval set prior to k-fold split
        eval_size : float
            Fraction of the training set to use for early stopping eval set
        eval_metric : str
            eval metric to use in early stopping
        """
        # Need pipe and X to fit
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert self.X is not None, 'X does not exist.  First set X.'
        
        # If no y provided, then no pipeline steps should require y
        step_list = [step[0] for step in self.pipe.steps]
        assert (('clf' not in step_list[-1])&('kf' not in step_list))|(self.y is not None), 'You must provide targets y if pipeline has a classifier step or feature selection step.'
        
        # Don't need to do a train-test split without a classifier
        assert (split_first==False)|('clf' in step_list[-1]), 'Only need train-test split if you have a classifier.'
                
        if split_first:
            X_train,X_val,y_train,y_val = train_test_split(self.X,self.y,stratify=self.y,
                                                           test_size=0.2,random_state=self.random_state)
            self.X_val = X_val
            self.y_val = y_val
        else:
            X_train = self.X.copy()
            if self.y is not None:
                y_train = self.y.copy()        
        
        # Restrict to features supplied in self.features
        X_train = X_train[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        
        # If XGB early stopping, then need to split off eval_set and define fit_params
        if isinstance(self.pipe[-1],XGBClassifier):
            if self.pipe[-1].get_params()['early_stopping_rounds'] is not None:
                X_train,X_es,y_train,y_es = train_test_split(X_train,y_train,
                                                               test_size=eval_size,
                                                               stratify=y_train,
                                                               random_state=self.random_state)
                trans_pipe = self.pipe[:-1]
                trans_pipe.fit_transform(X_train)
                X_es = trans_pipe.transform(X_es)
                clf_name = self.pipe.steps[-1][0]
                fit_params = {f'{clf_name}__eval_set':[(X_es,y_es)],
                              f'{clf_name}__eval_metric':eval_metric,
                             f'{clf_name}__verbose':0}
            else:
                fit_params = {}
        else:
            fit_params = {}
        
        # Fit and store fitted pipeline. If no classifier, fit_transform X_train and store transformed version
        pipe = self.pipe
        if 'clf' in step_list[-1]:
            pipe.fit(X_train,y_train,**fit_params)
        else:
            X_transformed = pipe.fit_transform(X_train)
            # X_transformed = pd.DataFrame(X_transformed,columns=pipe[-1].get_column_names_out())
            self.X_transformed = X_transformed
        self.pipe_fitted = pipe
    
    def predict_proba_pipeline(self, X_test = None):
        """
        Method for using a fitted pipeline to compute predicted
        probabilities for X_test (if supplied) or self.X_val
        Parameters:
        -----------
        X_test : pd.DataFrame or None
            test data input features (if None, will use self.X_val)
        """
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert 'clf' in self.pipe.steps[-1][0], 'The pipeline has no classifier.  Build a pipeline with a classifier first.'
        assert self.pipe_fitted is not None, 'Pipeline is not fitted.  First fit pipeline using fit_pipeline.'
        assert (X_test is not None)|(self.X_val is not None), 'Must either provide X_test and y_test or fit the pipeline with split_first=True.'
        
        if X_test is None:
            X_test = self.X_val
            
        # Restrict to features supplied in self.features
        X_test = X_test[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        
        # Save prediction
        self.y_predict_proba = self.pipe_fitted.predict_proba(X_test)[:,1]
        
    def score_pipeline(self,y_test=None,scoring='roc_auc',thresh=0.5, beta = 1,
                       normalize = None, print_score = True):
        """
        Method for scoring self.pipe_fitted on supplied test data and reporting score
        Parameters:
        -----------
        y_test : pd.Series or None
            true binary targets (if None, will use self.y_val)
        scoring : str
            specifies the metric to use for scoring
            must be one of
            'roc_auc', 'roc_plot', 'acc', 'f1', 'f1w', 'fb','mcc','kappa','conf','classif_report'
        thresh : float
            threshhold value for computing y_pred
            from y_predict_proba
        beta : float
            the beta parameter in the fb score
        normalize : str or None
            the normalize parameter for the 
            confusion_matrix. must be one of
            'true','pred','all',None
        print_score : bool
            if True, will print a message reporting the score
            if False, will return the score as a float
        """
        assert (y_test is not None)|(self.y_val is not None), 'Must either provide X_test and y_test or fit the pipeline with split_first=True.'
        assert self.y_predict_proba is not None, 'Predicted probabilities do not exist.  Run predict_proba_pipeline first.'
        
        if y_test is None:
            y_test = self.y_val
        
        # Score and report
        if scoring == 'roc_plot':
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            RocCurveDisplay.from_predictions(y_test,self.y_predict_proba,ax=ax)
            plt.show()
        elif scoring == 'roc_auc':
            score = roc_auc_score(y_test, self.y_predict_proba)
        else:
            y_pred = (self.y_predict_proba >= thresh).astype('int')
            if scoring == 'acc':
                score = accuracy_score(y_test,y_pred)
            elif scoring == 'f1':
                score = f1_score(y_test,y_pred)
            elif scoring == 'f1w':
                score = f1_score(y_test,y_pred,average='weighted')
            elif scoring == 'fb':
                score = fbeta_score(y_test,y_pred,beta=beta)
            elif scoring == 'mcc':
                score = matthews_coffcoeff(y_test,y_pred)
            elif scoring == 'kappa':
                score = cohen_kappa_score(y_test,y_pred)
            elif scoring == 'conf':
                fig = plt.figure(figsize=(3,3))
                ax = fig.add_subplot(111)
                ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=ax,colorbar=False)
                plt.show()
            elif scoring == 'classif_report':
                target_names=['neither seriously injured nor killed','seriously injured or killed']
                print(classification_report(y_test, y_pred,target_names=target_names))
            else:
                raise ValueError("scoring must be one of 'roc_auc', 'roc_plot','acc', 'f1', 'f1w', 'fb','mcc','kappa','conf','classif_report'")
        if scoring not in ['conf','roc_plot','classif_report']:
            if print_score:
                print(f'The {scoring} score is: {score}')
            else:
                return score
    
    def shap_values(self, X_test = None, eval_size=0.1,eval_metric='auc'):
        """
        Method for computing and SHAP values for features
        stratifiedtrain/test split
        A copy of self.pipe is fitted on the training set
        and then SHAP values are computed on test set samples
        Parameters:
        -----------
        X_test : pd.DataFrame
            The test set; if provided, will not perform
            a train/test split before fitting
        eval_size : float
            Fraction of the training set to use for early stopping eval set
        eval_metric : str
            eval metric to use in early stopping
        Returns: None (stores results in self.shap_vals)
        --------
        """
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert (self.X is not None)&(self.y is not None), 'X and/or y does not exist.  First supply X and y using set_data.'
        assert 'clf' in self.pipe.steps[-1][0], 'The pipeline has no classifier.  Build a pipeline with a classifier first.'
        
        
        # Clone pipeline, do train/test split if X_test not provided
        pipe = clone(self.pipe)
        X_train = self.X.copy()
        y_train = self.y.copy()
        if X_test is None:
            X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,stratify=y_train,
                                                                 test_size=0.2,random_state=self.random_state)
        # Restrict to features provided in self.features, and fit
        X_train = X_train[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        X_test = X_test[[feat for feat_type in self.features for feat in self.features[feat_type]]]
        
        # If XGB early stopping, then need to split off eval_set and define fit_params
        if isinstance(self.pipe[-1],XGBClassifier):
            if self.pipe[-1].get_params()['early_stopping_rounds'] is not None:
                X_train,X_es,y_train,y_es = train_test_split(X_train,y_train,
                                                               test_size=eval_size,
                                                               stratify=y_train,
                                                               random_state=self.random_state)
                trans_pipe = self.pipe[:-1]
                trans_pipe.fit_transform(X_train)
                X_es = trans_pipe.transform(X_es)
                clf_name = self.pipe.steps[-1][0]
                fit_params = {f'{clf_name}__eval_set':[(X_es,y_es)],
                              f'{clf_name}__eval_metric':eval_metric,
                             f'{clf_name}__verbose':0}
            else:
                fit_params = {}
        else:
            fit_params = {}
        
        pipe.fit(X_train,y_train,**fit_params)
            
        # SHAP will just explain classifier, so need transformed X_train and X_test
        X_train_trans, X_test_trans = pipe[:-1].transform(X_train), pipe[:-1].transform(X_test)
            
        # Need masker for linear model
        masker = shap.maskers.Independent(data=X_train_trans)
            
        # Initialize explainer and compute and store SHAP values as an explainer object
        explainer = shap.Explainer(pipe[-1], masker = masker, feature_names = pipe['col'].get_feature_names_out())
        self.shap_vals = explainer(X_test_trans)
        self.X_shap = X_train_trans
        self.y_shap = y_train
            
    def shap_plot(self,max_display='all'):
        """
        Method for generating plots of SHAP value results
        SHAP values should be already computed previously
        Generates two plots side by side:
            - a beeswarm plot of SHAP values of all samples
            - a barplot of mean absolute SHAP values
        Parameters:
        -----------
        max_display : int or 'all'
            The number of features to show in the plot, in descending
            order by mean absolute SHAP value.  If 'all', then
            all features will be included.
            
        Returns: None (plots displayed)
        --------
        """
        assert self.shap_vals is not None, 'No shap values exist.  First compute shap values.'
        assert (isinstance(max_display,int))|(max_display=='all'), "'max_display' must be 'all' or an integer"
        
        if max_display=='all':
            title_add = ', all features'
            max_display = self.shap_vals.shape[1]
        else:
            title_add = f', top {max_display} features'
            
        # Plot
        fig=plt.figure()
        ax1=fig.add_subplot(121)
        shap.summary_plot(self.shap_vals,plot_type='bar',max_display=max_display,
                          show=False,plot_size=0.2)
        ax2=fig.add_subplot(122)
        shap.summary_plot(self.shap_vals,plot_type='violin',max_display=max_display,
                          show=False,plot_size=0.2)
        fig.set_size_inches(12,max_display/3)
        
        ax1.set_title(f'Mean absolute SHAP values'+title_add,fontsize='small')
        ax1.set_xlabel('mean(|SHAP value|)',fontsize='x-small')
        ax2.set_title(f'SHAP values'+title_add,fontsize='small')
        ax2.set_xlabel('SHAP value', fontsize='x-small')
        for ax in [ax1,ax2]:
            ax.set_ylabel('feature name',fontsize='x-small')
            ax.tick_params(axis='y', labelsize='xx-small')
        plt.tight_layout()
        plt.show()
    
    def find_best_threshold(self,beta=1,conf=True,report=True, print_result=True):
        """
        Computes the classification threshold which gives the
        best F_beta score from classifier predictions,
        prints the best threshold and the corresponding F_beta score,
        and displays a confusion matrix and classification report
        corresponding to that threshold

        Parameters:
        -----------
        beta : float
            the desired beta value in the F_beta score
        conf : bool
            whether to display confusion matrix
        report : bool
            whether to display classification report
        print_result : bool
            whether to print a line reporting the best threshold
            and resulting F_beta score
        
        Returns: None (prints results and stores self.best_thresh)
        --------
        """
        prec,rec,threshs = precision_recall_curve(self.y_val,
                                                  self.y_predict_proba)
        F_betas = (1+beta**2)*(prec*rec)/((beta**2*prec)+rec)
        # Above formula is valid when TP!=0.  When TP==0
        # it gives np.nan whereas F_beta should be 0
        F_betas = np.nan_to_num(F_betas)
        idx = np.argmax(F_betas)
        best_thresh = threshs[idx]
        if print_result:
            print(f'Threshold optimizing F_{beta} score:   {best_thresh}\nBest F_{beta} score:   {F_betas[idx]}')
        if conf:
            self.score_pipeline(scoring='conf',thresh=best_thresh,beta=beta)
        if report:
            self.score_pipeline(scoring='classif_report',thresh=best_thresh,beta=beta)
        self.best_thresh = best_thresh

class LRStudy(ClassifierStudy):
    """
    A child class of ClassifierStudy which has an additional method specific to logistic regression
    """
    def __init__(self, classifier=None, X = None, y = None,
                 features=None,classifier_name = 'LR',
                 random_state=42):
        super().__init__(classifier, X, y,features,classifier_name,random_state)
    
    def plot_coeff(self, print_score = True, print_zero = False, title_add=None):
        """
        Method for doing a train/validation split, fitting the classifier,
        predicting and scoring on the validation set, and plotting
        a bar chart of the logistic regression coefficients corresponding
        to various model features.
        Features with coefficient zero and periodic spline features
        will be excluded from the chart.
        Parameters:
        -----------
        print_score : bool
            if True, the validation score are printed
        print_zero : bool
            if True, the list of features with zero coefficients are printed
        title_add : str or None
            an addendum that is added to the end of the plot title
        """
        assert self.pipe is not None, 'No pipeline exists; first build a pipeline using build_pipeline.'
        assert isinstance(self.classifier,LogisticRegression),'Your classifier is not an instance of Logistic Regression.'
        
        # fit and score
        self.fit_pipeline(split_first = True)
        self.predict_proba_pipeline()
        score = roc_auc_score(self.y_val, self.y_predict_proba)
        
        # Retrieve coeff values from fitted pipeline
        coeff = pd.DataFrame({'feature name':self.pipe_fitted['col'].get_feature_names_out(),
                               'coeff value':self.pipe_fitted[-1].coef_.reshape(-1)})\
                            .sort_values(by='coeff value')
        coeff = coeff[~coeff['feature name']\
                .isin([f'HOUR_OF_DAY_sp_{n}' for n in range(12)]\
                        +[f'DAY_OF_WEEK_sp_{n}' for n in range(3)])]\
                .set_index('feature name')
        coeff_zero_features = coeff[coeff['coeff value']==0].index
        coeff = coeff[coeff['coeff value']!=0]
        
        # Plot feature coefficients
        fig = plt.figure(figsize=(30,4))
        ax = fig.add_subplot(111)
        coeff['coeff value'].plot(kind='bar',ylabel='coeff value',ax=ax)
        ax.axhline(y=0, color= 'red', linewidth=2,)
        plot_title = 'PA bicycle collisions, 2002-2021\nLogistic regression model log-odds coefficients'
        if title_add is not None:
            plot_title += f': {title_add}'
        ax.set_title(plot_title)
        ax.tick_params(axis='x', labelsize='x-small')
        plt.show()
        
        if print_score:
            print(f'Score on validation set: {score}')
        if print_zero:
            print(f'Features with zero coefficients in trained model: {list(coeff_zero)}')
        
        self.score = score
        self.coeff = coeff
        self.coeff_zero_features = coeff_zero_features
        
            