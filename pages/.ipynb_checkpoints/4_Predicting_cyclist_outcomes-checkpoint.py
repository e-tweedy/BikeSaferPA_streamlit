import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.style.use('fivethirtyeight')
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth',None)
import scipy.stats as ss
import shap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from lib.study_classif import ClassifierStudy
import pickle

st.sidebar.title('BikeSaferPA suite')

st.header('BikeSaferPA visualization suite')
st.subheader('Predicting cyclist outcome with BikeSaferPA')

st.markdown("""
An instance of the BikeSaferPA predictice model has been trained in the background on all cyclist samples in the PENNDOT dataset.

Expand the following sections to adjust the factors in a hypothetical cyclist crash, and the model will provide a predicted probability that the cyclist involved suffers serious injury or fatality.
""")

########
#### Model pipeline trained in advance and dumped to pickle file
#######
# @st.cache_data
# def get_data(filename):
#     return pd.read_csv(filename)

# cyclists = get_data('cyclists.csv')

# # Define additional binary features
# cyclists['HILL'] = cyclists.GRADE.isin(['downhill','uphill','bottom_hill','top_hill']).astype(int)
# cyclists['FEMALE'] = cyclists['SEX'].replace({'F':1,'M':0})

# #Prepare input and target data
# features = {'cyc': ['DAY_OF_WEEK', 'HOUR_OF_DAY'],
#              'ord': ['HEAVY_TRUCK_COUNT','COMM_VEH_COUNT','SMALL_TRUCK_COUNT',
#                      'SUV_COUNT','VAN_COUNT'],
#              'cat': ['RESTRAINT_HELMET','VEH_ROLE','URBAN_RURAL',
#                      'ILLUMINATION','COLLISION_TYPE','IMPACT_SIDE',
#                     'GRADE','TCD_TYPE'],
#              'group': ['MUNICIPALITY', 'COUNTY', 'CRASH_MONTH'],
#              'num': ['AGE', 'SPEED_LIMIT', 'CRASH_YEAR'],
#              'bin': ['FEMALE','NON_INTERSECTION',
#                      'CURVED_ROAD','CURVE_DVR_ERROR',
#                      'DRINKING_DRIVER','DRUGGED_DRIVER',
#                      'AGGRESSIVE_DRIVING','LANE_DEPARTURE','NO_CLEARANCE',
#                      'NHTSA_AGG_DRIVING','CROSS_MEDIAN','RUNNING_RED_LT',
#                      'RUNNING_STOP_SIGN','TAILGATING','SPEEDING_RELATED',
#                      'MATURE_DRIVER','YOUNG_DRIVER',
#                     ]}
# TARGET = 'SERIOUS_OR_FATALITY'

# X,y = cyclists[sum(features.values(),[])],cyclists[TARGET]

# # Initialize and fit the classifier pipeline
# params={'l2_regularization': 2.4238734679222236,
#          'learning_rate': 0.14182851952262968,
#          'max_depth': 2,
#          'min_samples_leaf': 140}
# clf = HistGradientBoostingClassifier(early_stopping=True,max_iter=2000,
#                                      n_iter_no_change=50,random_state=42,
#                                     **params,class_weight='balanced')
# study = ClassifierStudy(clf,X,y,features=features)
# study.build_pipeline(cyc_method=None,num_ss=False)
# study.fit_pipeline()

# # Dump to pickle
filename = 'study.pkl'
# pickle.dump(study, open(filename, 'wb'))
###############

# Load trained pipeline from pickle
study = pickle.load(open(filename, 'rb'))

# Helper objects for collecting input settings
cat_data = {'ILLUMINATION':['daylight','dark_unlit','dark_lit','dusk','dawn'],
           'URBAN_RURAL':['urban','rural','urbanized'],
           'VEH_ROLE':['striking','struck','striking_struck'],
           'IMPACT_SIDE':['front','front_right','right','rear_right',
                         'rear','rear_left','left','front_left'],
           'RESTRAINT_HELMET':['bicycle_helmet','motorcycle_helmet',
                              'helmet_improper','no_restraint'],
           'COLLISION_TYPE':['angle','sideswipe_same_dir','sideswipe_opp_dir',
                            'head_on','rear_end','hit_ped'],
           'GRADE':['level','downhill','uphill','bottom_of_hill','top_of_hill'],
           'TCD_TYPE':['none','stop_sign','traffic_signal',
                         'flashing_traffic_signal','yield_sign']}

bin_data = [{'drug':['at least one drugged driver','DRUGGED_DRIVER'],
             'drink':['at least one drinking driver','DRINKING_DRIVER'],
            'speed':['at least one driver speeding','SPEEDING_RELATED'],
            'agg':['at least one aggressive driver','AGGRESSIVE_DRIVING'],
             'nhtsa':['at least one driver meeting NHTSA aggressive driving standard',
                      'NHTSA_AGG_DRIVING'],
            'red':['at least one driver running red light','RUNNING_RED_LT'],
            'stop':['at least one driver running stop sign','RUNNING_STOP_SIGN'],
             'clearance':['at least one driving proceeding without clearance from a stop',
                          'NO_CLEARANCE']},
             {'tail':['at least one driver tailgating','TAILGATING'],
             'curve_error':['at least one driver made error navigating curve',
                            'CURVE_DVR_ERROR'],
             'lane':['at least one driver departed their lane','LANE_DEPARTURE'],
             'median':['at least one driver crossed a median','CROSS_MEDIAN'],
             'curve':['crash was on a curved roadway','CURVED_ROAD'],
             'midblock':['crash occured midblock','NON_INTERSECTION'],
             'mature':['at least one driver over 65yrs','MATURE_DRIVER'],
             'young':['at least one driver under 20yrs','YOUNG_DRIVER']
            }]

veh_data = [('SUV','SUV'),
            ('heavy truck','HEAVY_TRUCK'),
            ('small truck','SMALL_TRUCK'),
            ('van','VAN'),
            ('commercial vehicle','COMM_VEH')]

# Initialize input sample and fill with user inputs
sample = pd.DataFrame(columns = study.pipe['col'].feature_names_in_)

with st.expander('Click here to expand or collapse numerical features'):
    cols = st.columns(3)
    with cols[0]:
        sample.loc[0,'AGE'] = st.number_input('Cyclist age (yrs):',
                            min_value=0,step=1,value=30)
        sample.loc[0,'SPEED_LIMIT'] = st.number_input('Posted speed limit (mph):',
                        min_value=0,max_value=100,step=5,value=25)
        sample.loc[0,'CRASH_YEAR'] = st.number_input('Year crash took place:',
                        min_value=2002,max_value=2023,step=1)
    with cols[1]:
        for k in [0,1,2]:
            sample.loc[0,f'{veh_data[k][1]}_COUNT']=st.number_input(f'# {veh_data[k][0]}s involved:',
                                                                    min_value=0,step=1,max_value=3)
    with cols[2]:
        for k in [3,4]:
            sample.loc[0,f'{veh_data[k][1]}_COUNT']=st.number_input(f'# {veh_data[k][0]}s involved:',
                                                                    min_value=0,step=1,max_value=3)
with st.expander('Click here to expand or collapse categorical features'):
    cols = st.columns(3)
    with cols[0]:
        sample.loc[0,'ILLUMINATION'] = st.selectbox('Illumination status:',
                                                      cat_data['ILLUMINATION'],
                                                      format_func= lambda x:x.replace('_',' and '))
        sample.loc[0,'URBAN_RURAL'] = st.selectbox('Collision setting:',cat_data['URBAN_RURAL'])
        sample.loc[0,'TCD_TYPE'] = st.selectbox('Traffic control device:',
                                                cat_data['TCD_TYPE'],
                                                format_func= lambda x:x.replace('_',' '))
    with cols[1]:
        sample.loc[0,'VEH_ROLE'] = st.selectbox('Bicycle role in collision:',
                                                cat_data['VEH_ROLE'],
                                                format_func= lambda x:x.replace('_',' and '))
        sample.loc[0,'IMPACT_SIDE'] = st.selectbox('Bicycle impact side:',
                                                   cat_data['IMPACT_SIDE'],
                                                   format_func= lambda x:x.replace('_',' '))
        sample.loc[0,'GRADE'] = st.selectbox('Roadway grade:',
                                             cat_data['GRADE'],
                                             format_func= lambda x:x.replace('_',' '))
    with cols[2]:
        sample.loc[0,'RESTRAINT_HELMET'] = st.selectbox('Cyclist helmet status:',
                                                        cat_data['RESTRAINT_HELMET'],
                                                        format_func= lambda x:x.replace('_',' ')\
                                                                   .replace('restraint','helmet'))
        sample.loc[0,'COLLISION_TYPE'] = st.selectbox('Collision type:',
                                                      cat_data['COLLISION_TYPE'],
                                                      format_func= lambda x:x.replace('_',' ')\
                                                                   .replace('dir','direction'))
        sample.loc[0,'FEMALE'] = st.selectbox('Cyclist identifies as female?',[0,1],
                                             format_func = lambda x:'yes' if x==1 else 'no')
    
with st.expander('Click here to expand or collapse binary features'):
    cols = st.columns(len(bin_data))
    for k,col in enumerate(cols):
        with col:
            for feat in bin_data[k]:
                sample.loc[0,bin_data[k][feat][1]]=int(st.checkbox(bin_data[k][feat][0],key=feat))

# Fill these arbitrarily - they won't influence inference
for feat in ['HOUR_OF_DAY','DAY_OF_WEEK','CRASH_MONTH','COUNTY','MUNICIPALITY']:
    sample.loc[0,feat]=1

# Predict and report result
study.predict_proba_pipeline(X_test=sample)

st.write(f'BikeSaferPA predicts a :red[{100*float(study.y_predict_proba):.2f}%] probability that a cyclist suffers serious injury or fatality under these conditions.')

st.subheader('SHAP analysis for this hypothetical prediction')

st.markdown("""
SHAP (SHapley Additive exPlainer) values provide an excellent method for assessing how various input features influence a model's predictions.  One significant advantage is that SHAP values are 'model agnostic' - they effectively explain the predictions made by many different types of machine learning classifiers, including the gradient boosted decision tree model used in BikeSaferPA.

The following 'force plot' shows the influence of each feature's SHAP value on the model's predicted probability that the cyclist suffers serious injury or fatality. A feature with a positive (resp. negative) SHAP value indicates that the feature's value pushes the predicted probability higher (resp. lower), which in the force plot corresponds to a push to the right (resp. left).  SHAP values can be interpreted as having units in the 'log-odds' space.

The force plot will update as you adjust input features in the menu above.
""")

# SHAP will just explain classifier, so need transformed X_train and X_test
pipe = study.pipe_fitted
X_trans, sample_trans = pipe[:-1].transform(study.X), pipe[:-1].transform(sample)
            
# # Need masker for linear model
# masker = shap.maskers.Independent(data=X_train_trans)
            
# Initialize explainer and compute and store SHAP values as an explainer object
explainer = shap.TreeExplainer(pipe[-1], feature_names = pipe['col'].get_feature_names_out())
shap_values = explainer(sample_trans)
sample_trans = pd.DataFrame(sample_trans,columns=pipe['col'].get_feature_names_out())

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

fig=shap.plots.force(explainer.expected_value,shap_values.values,sample_trans,figsize=(20,3),show=False,matplotlib=True)
st.pyplot(fig)