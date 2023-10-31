import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
import pickle
import shap
import lightgbm as lgb
from lightgbm import LGBMClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier

########################
### Helper functions ###
########################

@st.cache_data
def get_data(filename):
    """
    Read dataframe from CSV
    """
    return pd.read_csv(filename)


#############
### Setup ###
#############

# Load dataframes
crashes = get_data('crashes.csv')
cyclists = get_data('cyclists.csv')

# Load in prepared labeling data for app components
with open('app_data.pickle', 'rb') as file:
    period_data, cohort_data, time_cat_data, time_bin_data,\
    geo_data, county_data, feature_names, ord_features,\
    cat_features,flag_features,model_cat_data,\
    model_bin_data,veh_data = pickle.load(file)

features = cat_features+flag_features+ord_features
features.sort(key=lambda x:feature_names[x].lower())

# Load trained classifier study object
@st.cache_resource(show_spinner=False)
def load_study():
    """
    Load the trained classifier pipeline
    """
    with open('study.pkl', 'rb') as file:
        return pickle.load(file)
    



################################
### Initialize app structure ###
################################

st.header('BikeSaferPA: understanding cyclist outcomes')
tabs = st.tabs([
    'Welcome',
    'Crashes over time',
    'Mapping crashes',
    'Feature distributions',
    'BikeSaferPA predictions',
])
with tabs[0]:
    intro_container = st.container()
with tabs[1]:
    time_intro_container = st.container()
    time_settings_container = st.container()
    time_plot_container = st.container()
with tabs[2]:
    map_intro_container = st.container()
    map_settings_container = st.container()
    map_plot_container = st.container()
with tabs[3]:
    feature_intro_container = st.container()
    feature_settings_container = st.container()
    feature_plot_container = st.container()
with tabs[4]:
    model_intro_container = st.container()
    model_settings_container = st.container()
    model_result_container = st.container()
    model_shap_container = st.container()

############################
### Populate welcome tab ###
############################

with intro_container:
    st.markdown(
"""
This app provides a suite of tools to accompany Eamonn Tweedy's [BikeSaferPA project](https://github.com/e-tweedy/BikeSaferPA). These tools allow the user to:
- Visualize data related to crashes involving bicycles in Pennsylvania during the years 2002-2021, which was collected from a publically available [PENNDOT crash dataset](https://pennshare.maps.arcgis.com/apps/webappviewer/index.html?id=8fdbf046e36e41649bbfd9d7dd7c7e7e).
- Experiment with the BikeSaferPA model, which was trained on this cyclist crash data and designed to predict severity outcomes for cyclists based on crash data.

Navigate the tabs using the menu at the top to try them out.
    """)

######################################
### Populate crashes over time tab ###
######################################

### Intro text ###

with time_intro_container:
    st.subheader('Visualizing bicycle crashes in PA over time')

    st.markdown("""
This tool provides plots of cyclist crash counts by year, month of the year, day of the week, or hour of the day and can stratify the counts by various crash features.

You also have the option to restrict to Philadelpha county only, or the PA counties in the greater Philadelphia area (Bucks, Chester, Delaware, Montgomery, and Philadelphia).

Expand the toolbox below to choose plot options.
    """)

# Copy dataframe for this tab
crashes_time = crashes.copy()

### User input - settings for plot ###

with time_settings_container:
    # Expander containing plot option user input
    with st.expander('Click here to expand or collapse plot options menu'):
        col1,col2 = st.columns([0.4,0.6])
        with col1:
            # Geographic restriction selectbox
            geo = st.selectbox(
                'Geographic scope:',
                list(geo_data.keys()),index=0,
                format_func = lambda x:geo_data[x][0],
                key = 'time_geo_select',
            )
            # Time period selectbox
            period = st.selectbox(
                'Time period:',
                list(period_data.keys()),index=3,
                format_func = lambda x:period_data[x][0],
                key = 'time_period_select',
            )
    
        with col2:
            # Cyclist cohort selectbox
            cohort = st.selectbox(
                'Crash severity:',
                list(cohort_data.keys()),index=0,
                format_func = lambda x:cohort_data[x],
                key = 'time_cohort_select',
            )
            # Category stratification selectbox
            stratify = st.selectbox('Stratify crashes by:',
                                    ['no']+list(time_cat_data.keys()),index=0,
                                    key = 'time_cat_stratify_select',
                                    format_func = lambda x:time_cat_data[x][0]\
                                        if x!='no' else 'do not stratify',
                                   )
        st.markdown('Restrict to crashes containing the following factor(s):')
        title_add = ''
    
        cols = st.columns(len(time_bin_data))
        # Columns of binary feature checkboxes
        for k,col in enumerate(cols):
            with col:
                for feat in time_bin_data[k]:
                    # make checkbox
                    time_bin_data[k][feat][2]=st.checkbox(time_bin_data[k][feat][0],key=f'time_{feat}')
                    # if checked, filter samples and add feature to plot title addendum
                    if time_bin_data[k][feat][2]:
                        crashes_time = crashes_time[crashes_time[time_bin_data[k][feat][1]]==1]
                        title_add+= ', '+time_bin_data[k][feat][0].split('one ')[-1]

### Post-process user-selected setting data ###
                        

# Geographic restriction
if geo != 'statewide':
    crashes_time[crashes_time.COUNTY.isin(geo_data[geo][1])]
# Relegate rare categories to 'other' for plot readability
if stratify=='int_type':
    crashes_time['INTERSECT_TYPE']=crashes_time['INTERSECT_TYPE']\
    .replace({cat:'other' for cat in crashes_time.INTERSECT_TYPE.value_counts().index[3:]})
if stratify=='coll_type':
    crashes_time['COLLISION_TYPE']=crashes_time['COLLISION_TYPE']\
    .replace({cat:'other' for cat in crashes_time.COLLISION_TYPE.value_counts().index[6:]})
if stratify=='weather':
    crashes_time['WEATHER']=crashes_time['WEATHER']\
    .replace({cat:'other' for cat in crashes_time.WEATHER.value_counts().index[5:]})
if stratify=='tcd':
    crashes_time['TCD_TYPE']=crashes_time['TCD_TYPE']\
    .replace({cat:'other' for cat in crashes_time.TCD_TYPE.value_counts().index[3:]})
crashes_time=crashes_time.dropna(subset=period_data[period][1])

# Order categories in descending order by frequency
category_orders = {time_cat_data[cat][1]:list(crashes_time[time_cat_data[cat][1]].value_counts().index) for cat in time_cat_data}

# Define cohort
if cohort == 'inj':
    crashes_time = crashes_time[crashes_time.BICYCLE_SUSP_SERIOUS_INJ_COUNT > 0]
elif cohort == 'fat':
    crashes_time = crashes_time[crashes_time.BICYCLE_DEATH_COUNT > 0]

# Replace day,month numbers with string labels
if period in ['day','month']:
    crashes_time[period_data[period][1]] = crashes_time[period_data[period][1]].apply(lambda x:period_data[period][2][x-1])

# Plot title addendum
if len(title_add)>0:
    title_add = '<br>with'+title_add.lstrip(',')

# Category stratification plot settings
if stratify=='no':
    color,legend_title = None,None
else:
    color,legend_title=time_cat_data[stratify][1],time_cat_data[stratify][2]
    title_add += f'<br>stratified {time_cat_data[stratify][0]}'

### Build and display plot ###

with time_plot_container:
    # Plot samples if any, else report no samples remain
    if crashes_time.shape[0]>0:
        fig = px.histogram(crashes_time, 
                           x=period_data[period][1],
                           color=color,
                           nbins=len(period_data[period][2]),
                           title=f'PA bicycle crashes 2002-2021 by {period_data[period][0]} - {cohort_data[cohort]}'+title_add,
                           category_orders = category_orders,
                          )
        fig.update_layout(bargap=0.2,
                          xaxis_title=period_data[period][0],
                          legend_title_text=legend_title,
                         )
        fig.update_xaxes(categoryorder="array",
                         categoryarray=period_data[period][2],
                         dtick=1,
                        )
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.markdown('#### No samples meet these criteria. Please remove some factors.')

####################################
### Populate mapping crashes tab ###
####################################

### Intro text ###

with map_intro_container:
    st.subheader('Mapping bicycle crashes in PA')

    st.markdown("""
This tool provides interactive maps of crash events, either statewide or in one of the more populous counties.  Crash event dots are color-coded based on whether the crash involved serious cyclist injury, cyclist fatality, or neither.

Expand the menu below to adjust map options.
    """)

# Copy dataframe for this tab
crashes_map = crashes.copy()

### User input - settings for map plot ###

with map_settings_container:
    # Expander containing plot option user input
    with st.expander('Click here to expand or collapse map options menu'):
        # Locale selectbox
        geo = st.selectbox(
            'Select either statewide or a particular county to plot:',
            ['Statewide']+[county+' County' for county in county_data],
            key = 'map_geo_select',
        )
        # Animation status selectbox
        animate = st.selectbox(
            'Select how to animate the map:',
            ['do not animate','by year','by month'],
            key = 'map_animate_select',
        )

### Post-process user-selected setting data ###

if geo == 'Statewide':
    county = None
else:
    geo = geo.split(' ')[0]
    county = (county_data[geo],geo)
color_dots=True
if animate == 'do not animate':
    animate = False
    animate_by=None
else:
    animate_by = animate.split(' ')[1]
    animate = True
    # If county is not None and animating, check whether first frame has all
    # injury/fatality status categories.  If not, then we will not color dots
    # by injury/fatality status.
    # This is to account for bug/feature in plotly 'animation_frame' and 'color' functionality
    # which yields unexpected results when all color categories not present in first frame
    # see e.g. https://github.com/plotly/plotly.py/issues/2259
    
    if county is not None:
        if animate_by == 'year':
            color_dots = len(crashes_map.query('COUNTY==@county[0] and CRASH_YEAR==2002')\
                       .BICYCLE_DEATH_COUNT.unique())+\
                        len(crashes_map.query('COUNTY==@county[0] and CRASH_YEAR==2002')\
                       .BICYCLE_SUSP_SERIOUS_INJ_COUNT.unique()) > 3
        else:
            color_dots = len(crashes_map.query('COUNTY==@county[0] and CRASH_YEAR==2002 and CRASH_MONTH==1')\
                       .BICYCLE_DEATH_COUNT.unique())+\
                       len(crashes_map.query('COUNTY==@county[0] and CRASH_YEAR==2002 and CRASH_MONTH==1')\
                       .BICYCLE_SUSP_SERIOUS_INJ_COUNT.unique()) > 3
if color_dots==False:
    st.markdown("""
    **Warning:** color-coding by injury/death status is disabled; this feature gives unexpected results
    when not all classes appear in the first animation frame due to bug/feature in Plotly animate functionality.
    Injury/death status is still visible in hover-text box.
    """)

### Build and display map plot ###

from lib.vis_data import plot_map

with map_plot_container:
    fig = plot_map(
        df=crashes_map,county=county,animate=animate,
        color_dots=color_dots,animate_by=animate_by,
        show_fig=False,return_fig=True,
    )
    st.plotly_chart(fig,use_container_width=True)

##########################################
### Populate feature distributions tab ###
##########################################

### Intro text ###

with feature_intro_container:
    st.subheader('Visualizing crash feature distributions')

    st.markdown("""
The tools on this page will demonstrate how distributions of values of various crash and cyclist features vary between two groups:
- all cyclists involved in crashes, and
- those cyclists who suffered serious injury or fatality

Expand the following menu to choose a feature, and the graph will show its distribution of its values (via percentages) over the two groups.  Again you may restrict to Philadelpha county only, or the PA counties in the greater Philadelphia area (Bucks, Chester, Delaware, Montgomery, and Philadelphia).

Pay particular attention to feature values which become more or less prevalent among cyclists suffering serious injury or death - for instance, 6.2% of all cyclists statewide were involved in a head-on collision, whereas 11.8% of those with serious injury or fatality were in a head-on collision.
    """)

# Copy dataframe for this tab
cyclists_feat = cyclists.copy()

### User input - settings for plot ###

with feature_settings_container:
    # Expander containing plot option user input
    with st.expander('Click here to expand or collapse feature selection menu'):
        # Geographic restriction selectbox
        geo = st.selectbox(
            'Geographic scope:',
            list(geo_data.keys()),index=0,
            format_func = lambda x:geo_data[x][0],
            key = 'feature_geo_select',
        )
        # Feature selectbox
        feature = st.selectbox('Show distributions of this feature:',
                               features,format_func = lambda x:feature_names[x],
                               key = 'feature_select',
                              )

### Post-process user-selected settings data ###

from lib.vis_data import feat_perc,feat_perc_bar
# Geographic restriction
if geo != 'statewide':
    cyclists_feat = cyclists_feat[cyclists_feat.COUNTY.isin(geo_data[geo][1])]
    
# Recast binary and day of week data
if feature not in ord_features:
    cyclists_feat[feature]=cyclists_feat[feature].replace({1:'yes',0:'no'})
if feature == 'DAY_OF_WEEK':
    cyclists_feat[feature]=cyclists_feat[feature].astype(str)

### Build and display plot ###

with feature_plot_container:
    
    # Generate plot
    sort = False if feature in ord_features else True
    fig = feat_perc_bar(
        feature,cyclists_feat, feat_name=feature_names[feature],
        return_fig=True,show_fig=False,sort=sort
    )

    # Adjust some colorscale and display settings
    if feature == 'SPEED_LIMIT':
        fig.update_coloraxes(colorscale='YlOrRd',cmid=35)
    if feature == 'HOUR_OF_DAY':
        fig.update_coloraxes(colorscale='balance')
    if feature == 'DAY_OF_WEEK':
        days = ['Sun']+list(cal.day_abbr)[:-1]
        for idx, day in enumerate(days):
            fig.data[idx].name = day
            fig.data[idx].hovertemplate = day

    # Display plot
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('See [this Jupyter notebook](https://e-tweedy.github.io/2_BikeSaferPA_vis.html) for an in-depth data exploration and visualization process.')

######################################
### Populate model predictions tab ###
######################################

from lib.study_classif import ClassifierStudy
    
### Intro text ###

with model_intro_container:
    st.subheader('Predicting cyclist outcome with BikeSaferPA')

    st.markdown("""
An instance of the BikeSaferPA predictive model has been trained in advance on all cyclist samples in the PENNDOT dataset.  This model is a gradient-boosted decision tree classifier model, and the model selection and evaluation process is covered in detail in [this Jupyter notebook](https://e-tweedy.github.io/3_BikeSaferPA_models.html).

The purpose of this tool is to allow the user to simulate a model prediction on a hypothetical sample, and then explain the model's prediction using SHAP values.

Expand the following sections to adjust the factors in a hypothetical cyclist crash, and the model will provide a predicted probability that the cyclist involved suffers serious injury or fatality.  You'll find that some factors influence the prediction significantly, and others very little.
    """)

### User inputs for model prediction ###

# Load the trained classifier study object
study = load_study()
    
# Initialize input sample.  User inputs will update values.
sample = pd.DataFrame(columns = study.pipe['col'].feature_names_in_)

with model_settings_container:
    # Expander for numerical inputs
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
                sample.loc[0,f'{veh_data[k][1]}_COUNT']=st.number_input(
                    f'# {veh_data[k][0]}s involved:',
                    min_value=0,step=1,max_value=3
                )
        with cols[2]:
            for k in [3,4]:
                sample.loc[0,f'{veh_data[k][1]}_COUNT']=st.number_input(
                    f'# {veh_data[k][0]}s involved:',
                    min_value=0,step=1,max_value=3
                )
    # Expander for categorical inputs
    with st.expander('Click here to expand or collapse categorical features'):
        cols = st.columns(3)
        with cols[0]:
            sample.loc[0,'ILLUMINATION'] = st.selectbox(
                'Illumination status:',
                model_cat_data['ILLUMINATION'],
                format_func= lambda x:x.replace('_',' and '),
            )
            sample.loc[0,'URBAN_RURAL'] = st.selectbox(
                'Collision setting:',
                model_cat_data['URBAN_RURAL'],
            )
            sample.loc[0,'TCD_TYPE'] = st.selectbox(
                'Traffic control device:',
                model_cat_data['TCD_TYPE'],
                format_func= lambda x:x.replace('_',' '),
            )
        with cols[1]:
            sample.loc[0,'VEH_ROLE'] = st.selectbox(
                'Bicycle role in collision:',
                model_cat_data['VEH_ROLE'],
                format_func= lambda x:x.replace('_',' and '),
            )
            sample.loc[0,'IMPACT_SIDE'] = st.selectbox(
                'Bicycle impact side:',
                model_cat_data['IMPACT_SIDE'],
                format_func= lambda x:x.replace('_',' '),
            )
            sample.loc[0,'GRADE'] = st.selectbox(
                'Roadway grade:',
                model_cat_data['GRADE'],
                format_func= lambda x:x.replace('_',' '),
            )
        with cols[2]:
            sample.loc[0,'RESTRAINT_HELMET'] = st.selectbox(
                'Cyclist helmet status:',
                model_cat_data['RESTRAINT_HELMET'],
                format_func= lambda x:x.replace('_',' ')\
                .replace('restraint','helmet'),
            )
            sample.loc[0,'COLLISION_TYPE'] = st.selectbox(
                'Collision type:',
                model_cat_data['COLLISION_TYPE'],
                format_func= lambda x:x.replace('_',' ')\
                .replace('dir','direction'),
            )
            sample.loc[0,'FEMALE'] = st.selectbox(
                'Cyclist sex:*',[1,0],
                format_func = lambda x:'F' if x==1 else 'M',
            )
            st.markdown('*Note: the PENNDOT dataset only has a binary sex feature.')

    # Expander for binary inputs
    with st.expander('Click here to expand or collapse binary features'):
        cols = st.columns(len(model_bin_data))
        for k,col in enumerate(cols):
            with col:
                for feat in model_bin_data[k]:
                    sample.loc[0,model_bin_data[k][feat][1]]=int(st.checkbox(model_bin_data[k][feat][0],
                                                                             key=f'model_{feat}'))

### Model prediction and reporting result ###

with model_result_container:
    # Fill these columns arbitrarily - they won't affect inference
    # COUNTY, MUNICIPALITY, HOUR_OF_DAY, CRASH_MONTH used in pipeline for NaN imputation
    # This version of model doesn't use temporal features as we set cyc_method=None
    for feat in ['HOUR_OF_DAY','DAY_OF_WEEK','CRASH_MONTH','COUNTY','MUNICIPALITY']:
        sample.loc[0,feat]=1

    # Predict and report result
    # study.predict_proba_pipeline(X_test=sample)
    feature_names = study.pipe_fitted[-2].get_feature_names_out()
    pipe = study.pipe_fitted
    sample_trans = pipe[:-1].transform(sample)
    y_predict_proba = pipe.predict_proba(sample)[0,1]

    st.write(f'**BikeSaferPA predicts a :red[{100*y_predict_proba:.2f}%] probability that a cyclist suffers serious injury or fatality under these conditions.**')

### SHAP values ####

with model_shap_container:
    st.subheader('SHAP analysis for this hypothetical prediction')

    st.markdown("""
SHAP (SHapley Additive exPlainer) values provide an excellent method for assessing how various input features influence a model's predictions.  One significant advantage is that SHAP values are 'model agnostic' - they effectively explain the predictions made by many different types of machine learning classifiers.

The following 'force plot' shows the influence of each feature's SHAP value on the model's predicted probability that the cyclist suffers serious injury or fatality. A feature with a positive (resp. negative) SHAP value indicates that the feature's value pushes the predicted probability higher (resp. lower), which in the force plot corresponds to a push to the right (resp. left).

The force plot will update as you adjust input features in the menu above.
    """)

    # SHAP will just explain classifier, so need transformed X_train and X_test
            
    # # Need masker for linear model
    # masker = shap.maskers.Independent(data=X_train_trans)
            
    # Initialize explainer and compute and store SHAP values as an explainer object
    # shap_values_list = []
    # for calibrated_classifier in clf.calibrated_classifiers_:
    #     explainer = shap.TreeExplainer(calibrated_classifier.estimator,feature_names = pipe['col'].get_feature_names_out())
    #     shap_values = explainer(sample_trans)
    #     shap_values_list.append(shap_values.values)
    # shap_values = np.array(shap_values_list).sum(axis=0) / len(shap_values_list)
    explainer = shap.TreeExplainer(pipe[-1], feature_names = pipe['col'].get_feature_names_out())
    shap_values = explainer(sample_trans)
    sample_trans = pd.DataFrame(sample_trans,columns=pipe['col'].get_feature_names_out())
    # def st_shap(plot, height=None):
    #     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    #     components.html(shap_html, height=height)
    fig=shap.plots.force(explainer.expected_value[1],shap_values.values[0][:,1],sample_trans,
                         figsize=(20,3),show=False,matplotlib=True)
    st.pyplot(fig)
