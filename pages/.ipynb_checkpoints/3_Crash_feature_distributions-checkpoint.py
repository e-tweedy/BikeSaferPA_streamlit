import streamlit as st
import pandas as pd
from lib.vis_data import feat_perc,feat_perc_bar
import calendar as cal

st.sidebar.title('BikeSaferPA suite')

st.header('BikeSaferPA: understanding cyclist outcomes')
st.subheader('Visualizing crash feature distributions')

st.markdown("""
The tools on this page will demonstrate how distributions of values of various crash and cyclist features vary between two groups:
- all cyclists involved in crashes, and
- those cyclists who suffered serious injury or fatality

Expand the following menu to choose a feature, and the graph will show its distribution of its values (via percentages) over the two groups.

Pay particular attention to feature values which become more or less prevalent among cyclists suffering serious injury or death - for instance, 6.2% of all cyclists were involved in a head-on collision, whereas 11.8% of those with serious injury or fatality were in a head-on collision.
""")
@st.cache_data
def get_data(filename):
    return pd.read_csv(filename)

cyclists = get_data('cyclists.csv')

feature_names = {'AGE_BINS':'cyclist age (binned)',
                 'SEX':'cyclist sex',
                 'RESTRAINT_HELMET':'cyclist helmet status',
                 'URBAN_RURAL':'crash setting',
                'VEH_ROLE':'cyclist striking or struck',
                 'VEH_MOVEMENT':'cyclist movement during collision',
                'IMPACT_SIDE':'cyclist impact side',
                'DAY_OF_WEEK':'day of the week',
                'HOUR_OF_DAY':'hour of the day',
                'COLLISION_TYPE':'collision type',
                'RDWY_ALIGNMENT':'roadway alignment',
                'GRADE':'roadway grade',
                'SPEED_LIMIT':'posted speed limit',
                'ROAD_CONDITION':'roadway condition',
                'WEATHER':'weather status',
                'ILLUMINATION':'illumination status',
                'INTERSECT_TYPE':'intersection type',
                'TCD_TYPE':'traffic control device type',
                'TCD_FUNC_CD':'traffic control device status',
                 'SPEEDING_RELATED':'speeding related',
                 'NHTSA_AGG_DRIVING':'aggressive driving involved (NHTSA standard)',
                 'RUNNING_RED_LT':'running red light involved',
                 'CROSS_MEDIAN':'driver crossed median',
                 'NO_CLEARANCE':'driver proceeded w/out clearance from stop',
                 'FATIGUE_ASLEEP':'driver was fatigued or asleep',
                 'DISTRACTED':'driver was distracted',
                 'CELL_PHONE':'driver was using cell phone'
                }
for string in ['BUS','HEAVY_TRUCK','SMALL_TRUCK','COMM_VEHICLE','SUV',
           'MATURE_DRIVER','YOUNG_DRIVER','IMPAIRED_DRIVER','AGGRESSIVE_DRIVING',
              'DRINKING_DRIVER','DRUGGED_DRIVER','LANE_DEPARTURE',
              'TAILGATING','RUNNING_STOP_SIGN',]:
    feature_names[string] = string.replace('_',' ').lower()+' involved'

ord_features = ['AGE_BINS','DAY_OF_WEEK','HOUR_OF_DAY','SPEED_LIMIT']
cat_features = ['SEX','RESTRAINT_HELMET',
                'VEH_ROLE','VEH_MOVEMENT',
                'IMPACT_SIDE','URBAN_RURAL',
                'COLLISION_TYPE','RDWY_ALIGNMENT',
                'GRADE','ROAD_CONDITION','WEATHER',
                'ILLUMINATION','INTERSECT_TYPE',
                'TCD_TYPE', 'TCD_FUNC_CD']
flag_features = ['BUS', 'HEAVY_TRUCK', 'SMALL_TRUCK', 'SUV','COMM_VEHICLE', 
                 'RUNNING_STOP_SIGN','RUNNING_RED_LT','SPEEDING_RELATED', 'TAILGATING',
                 'CROSS_MEDIAN', 'LANE_DEPARTURE','AGGRESSIVE_DRIVING','NHTSA_AGG_DRIVING',
                 'CELL_PHONE','DISTRACTED','DRINKING_DRIVER', 'DRUGGED_DRIVER',
                 'FATIGUE_ASLEEP','IMPAIRED_DRIVER',
                 'MATURE_DRIVER','YOUNG_DRIVER','NO_CLEARANCE']

features = cat_features+flag_features+ord_features
features.sort(key=lambda x:feature_names[x].lower())

with st.expander('Click here to expand or collapse feature selection menu'):
    feature = st.selectbox('Show distributions of this feature:',features,
                          format_func = lambda x:feature_names[x])
if feature not in ord_features:
    cyclists[feature]=cyclists[feature].replace({1:'yes',0:'no'})
if feature == 'DAY_OF_WEEK':
    cyclists[feature]=cyclists[feature].astype(str)
#     cyclists[feature]=cyclists[feature]\
#                     .replace({k+1:day for k,day in enumerate(['Sun']+list(cal.day_abbr)[:-1])})
sort = False if feature in ord_features else True
fig = feat_perc_bar(feature,cyclists, feat_name=feature_names[feature],
                    return_fig=True,show_fig=False,sort=sort)
if feature == 'SPEED_LIMIT':
    fig.update_coloraxes(colorscale='YlOrRd',cmid=35)
if feature == 'HOUR_OF_DAY':
    fig.update_coloraxes(colorscale='balance')
if feature == 'DAY_OF_WEEK':
    days = ['Sun']+list(cal.day_abbr)[:-1]
    for idx, day in enumerate(days):
        fig.data[idx].name = day
        fig.data[idx].hovertemplate = day
st.plotly_chart(fig,use_container_width=True)

st.markdown('See [this Jupyter notebook](https://e-tweedy.github.io/2_BikeSaferPA_vis.html) for a in-depth data exploration and visualization process.')