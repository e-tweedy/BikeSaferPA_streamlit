import pandas as pd
import numpy as np
import streamlit as st
import calendar as cal
import plotly.express as px

st.sidebar.title('BikeSaferPA suite')

# Get dataset
@st.cache_data
def get_data(filename):
    return pd.read_csv(filename)

crashes = get_data('crashes.csv')

st.header('BikeSaferPA: understanding cyclist outcomes')
st.subheader('Visualizing bicycle crashes in PA over time')

st.markdown("""
This tool provides plots of cyclist crash counts by year, month of the year, day of the week, or hour of the day and can stratify the counts by various crash features.

You also have the option to restrict to Philadelpha county only, or the PA counties in the greater Philadelphia area (Bucks, Chester, Delaware, Montgomery, and Philadelphia).

Expand the toolbox below to choose plot options.
""")

# Data used for input elements and plot elements
period_data = {'hour':('hour of the day','HOUR_OF_DAY',list(range(24))),
                'day':('day of the week','DAY_OF_WEEK',['Sun']+list(cal.day_abbr)[:-1]),
                'month':('month of the year','CRASH_MONTH',list(cal.month_abbr)[1:]),
                'year':('year','CRASH_YEAR',list(range(2002,2022)))}
cohort_data = {'all':'all crashes involving bicycles',
                'inj':'at least one serious cyclist injury',
                'fat':'at least one cyclist fatality'}
cat_data = {'urban':('by urban, rural, or urbanized setting','URBAN_RURAL','Crash setting'),
            'coll_type':('by collision type','COLLISION_TYPE','Collision type'),
            'int_type':('by intersection type','INTERSECT_TYPE','Intersection type'),
            'ill':('by illumination status','ILLUMINATION','Illumination status'),
            'weather':('by weather status','WEATHER','Weather status'),
            'tcd':('by traffic control device present','TCD_TYPE','Traffic control device')
              }
bin_data = [{'drink':['at least one drinking driver','DRINKING_DRIVER',False],
            'drug':['at least one drugged driver','DRUGGED_DRIVER',False],
            'speed':['at least one driver speeding','SPEEDING',False],
            'agg':['at least one aggressive driver','AGGRESSIVE_DRIVING',False],
            'red':['at least one driver running red light','RUNNING_RED_LT',False],
            'stop':['at least one driver running stop sign','RUNNING_STOP_SIGN',False],
            },
            {'suv':['at least one SUV','SUV',False],
            'ht':['at least one heavy truck','HEAVY_TRUCK',False],
            'st':['at least one small truck','SMALL_TRUCK',False],
            'com':['at least one commercial vehicle','COMM_VEHICLE',False],
            'bus':['at least one bus','BUS',False],
            'van':['at least one van','VAN',False]
            }]
geo_data = {'statewide':['All of Pennsylvania',None],
           'greater_phila':['Greater Philadelphia area',[67, 9, 15, 23, 46]],
           'phila':['Philadelphia County',[67]]}

# Expander containing plot option user input
with st.expander('Click here to expand or collapse plot options menu'):
    col1,col2 = st.columns([0.4,0.6])
    with col1:
        # Geographic restriction selectbox
        geo = st.selectbox(
            'Geographic scope:',
            list(geo_data.keys()),index=0,
            format_func = lambda x:geo_data[x][0])
        # Time period selectbox
        period = st.selectbox(
            'Time period:',
            list(period_data.keys()),index=3,
            format_func = lambda x:period_data[x][0])
    
    with col2:
        # Cyclist cohort selectbox
        cohort = st.selectbox(
            'Crash severity:',
            list(cohort_data.keys()),index=0,
            format_func = lambda x:cohort_data[x])
        # Category stratification selectbox
        stratify = st.selectbox('Stratify crashes by:',
                       ['no']+list(cat_data.keys()),index=0,
                       format_func = lambda x:cat_data[x][0] if x!='no' else 'do not stratify')
    st.markdown('Restrict to crashes containing the following factor(s):')
    title_add = ''
    
    cols = st.columns(len(bin_data))
    # Columns of binary feature checkboxes
    for k,col in enumerate(cols):
        with col:
            for feat in bin_data[k]:
                # make checkbox
                bin_data[k][feat][2]=st.checkbox(bin_data[k][feat][0],key=feat)
                # if checked, filter samples and add feature to plot title addendum
                if bin_data[k][feat][2]:
                    crashes = crashes[crashes[bin_data[k][feat][1]]==1]
                    title_add+= ', '+bin_data[k][feat][0].split('one ')[-1]

# Geographic restriction
if geo != 'statewide':
    crashes = crashes[crashes.COUNTY.isin(geo_data[geo][1])]
# Relegate rare categories to 'other' for plot readability
if stratify=='int_type':
    crashes['INTERSECT_TYPE']=crashes['INTERSECT_TYPE']\
    .replace({cat:'other' for cat in crashes.INTERSECT_TYPE.value_counts().index[3:]})
if stratify=='coll_type':
    crashes['COLLISION_TYPE']=crashes['COLLISION_TYPE']\
    .replace({cat:'other' for cat in crashes.COLLISION_TYPE.value_counts().index[6:]})
if stratify=='weather':
    crashes['WEATHER']=crashes['WEATHER']\
    .replace({cat:'other' for cat in crashes.WEATHER.value_counts().index[5:]})
if stratify=='tcd':
    crashes['TCD_TYPE']=crashes['TCD_TYPE']\
    .replace({cat:'other' for cat in crashes.TCD_TYPE.value_counts().index[3:]})
crashes=crashes.dropna(subset=period_data[period][1])

# Order categories in descending order by frequency
category_orders = {cat_data[cat][1]:list(crashes[cat_data[cat][1]].value_counts().index) for cat in cat_data}

# Define cohort
if cohort == 'inj':
    crashes = crashes[crashes.BICYCLE_SUSP_SERIOUS_INJ_COUNT > 0]
elif cohort == 'fat':
    crashes = crashes[crashes.BICYCLE_DEATH_COUNT > 0]

# Replace day,month numbers with string labels
if period in ['day','month']:
    crashes[period_data[period][1]] = crashes[period_data[period][1]].apply(lambda x:period_data[period][2][x-1])

# Plot title addendum
if len(title_add)>0:
    title_add = '<br>with'+title_add.lstrip(',')

# Category stratification plot settings
if stratify=='no':
    color,legend_title = None,None
else:
    color,legend_title=cat_data[stratify][1],cat_data[stratify][2]
    title_add += f'<br>stratified {cat_data[stratify][0]}'

# Plot samples if any, else report no samples remain
if crashes.shape[0]>0:
    fig = px.histogram(crashes, x=period_data[period][1],color=color,nbins=len(period_data[period][2]),
                      title=f'PA bicycle crashes 2002-2021 by {period_data[period][0]} - {cohort_data[cohort]}'\
                       +title_add,
                      category_orders = category_orders)
    fig.update_layout(bargap=0.2,
                     xaxis_title=period_data[period][0],
                     # legend_traceorder='reversed',
                     legend_title_text=legend_title)
    fig.update_xaxes(categoryorder="array",
                    categoryarray=period_data[period][2],
                    dtick=1)
    st.plotly_chart(fig,use_container_width=True)
else:
    st.markdown('#### No samples meet these criteria. Please remove some factors.')
