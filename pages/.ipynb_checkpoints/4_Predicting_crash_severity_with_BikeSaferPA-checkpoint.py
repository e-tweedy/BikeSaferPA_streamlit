import streamlit as st
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

st.sidebar.title('BikeSaferPA')

@st.cache_data
def get_data(filename):
    return pd.read_csv(filename)

cyclists = get_data('cyclists.csv')

# Define additional binary features
cyclists['HILL'] = cyclists.GRADE.isin(['downhill','uphill','bottom_hill','top_hill']).astype(int)
cyclists['FEMALE'] = cyclists['SEX'].replace({'F':1,'M':0})
# for tcd_type in ['traffic_signal','flashing_traffic_signal','stop_sign']:
#     cyclists[f'TCD_{tcd_type}'] = (cyclists.TCD_TYPE==tcd_type).astype(int)

#Prepare input and target data
features = {'cyc': ['DAY_OF_WEEK', 'HOUR_OF_DAY'],
             'ord': ['HEAVY_TRUCK_COUNT','COMM_VEH_COUNT','SMALL_TRUCK_COUNT',
                     'SUV_COUNT','VAN_COUNT'],
             'cat': ['RESTRAINT_HELMET','VEH_ROLE','URBAN_RURAL',
                     'ILLUMINATION','COLLISION_TYPE','IMPACT_SIDE',
                    'GRADE','TCD_TYPE'],
             'group': ['MUNICIPALITY', 'COUNTY', 'CRASH_MONTH'],
             'num': ['AGE', 'SPEED_LIMIT', 'CRASH_YEAR'],
             'bin': ['FEMALE','NON_INTERSECTION',
                     'CURVED_ROAD','CURVE_DVR_ERROR',
                     'DRINKING_DRIVER','DRUGGED_DRIVER',
                     'AGGRESSIVE_DRIVING','LANE_DEPARTURE','NO_CLEARANCE',
                     'NHTSA_AGG_DRIVING','CROSS_MEDIAN','RUNNING_RED_LT',
                     'RUNNING_STOP_SIGN','TAILGATING','SPEEDING_RELATED',
                     'MATURE_DRIVER','YOUNG_DRIVER',
                     # 'TCD_flashing_traffic_signal','TCD_stop_sign','TCD_traffic_signal',
                    ]}
TARGET = 'SERIOUS_OR_FATALITY'

X,y = cyclists[sum(features.values(),[])],cyclists[TARGET]
# X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,
#                                                     test_size=0.2,
#                                                     random_state=42)

# Initialize and fit the classifier pipeline
@st.cache_resource
def fit_model(_clf,X,y,features):
    study = ClassifierStudy(clf,X,y,features=features)
    study.build_pipeline(cyc_method=None,num_ss=False)
    study.fit_pipeline()
    st.write('Done fitting the model!')
    return study

params={'l2_regularization': 2.4238734679222236,
         'learning_rate': 0.14182851952262968,
         'max_depth': 2,
         'min_samples_leaf': 140}
clf = HistGradientBoostingClassifier(early_stopping=True,max_iter=2000,
                                     n_iter_no_change=50,random_state=42,
                                    **params,class_weight='balanced')
study = fit_model(clf,X,y,features)


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

sample = pd.DataFrame(columns = X.columns)

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
for feat in ['HOUR_OF_DAY','DAY_OF_WEEK','CRASH_MONTH','COUNTY','MUNICIPALITY']:
    sample.loc[0,feat]=1
study.predict_proba_pipeline(X_test=sample)

st.write(f'Probability that the cyclist is seriously injured or killed:  {100*float(study.y_predict_proba):.2f}%')
