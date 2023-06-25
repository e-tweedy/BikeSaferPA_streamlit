import pandas as pd
import numpy as np
import streamlit as st
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import seaborn as sns
# from IPython.display import display, display_html
import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from plotly.offline import plot
from scipy import stats

st.sidebar.title('BikeSaferPA suite')

from lib.vis_data import plot_map

@st.cache_data
def get_data(filename):
    return pd.read_csv(filename)

crashes = get_data('crashes.csv')

st.header('BikeSaferPA visualization suite')
st.subheader('Mapping bicycle crashes in PA')

county_data = {'Philadelphia':67,'Allegheny':2,
               'Montgomery':46,'Bucks':9,
               'Delaware':23,'Lancaster':36,
               'Chester':15,'York':66,
               'Berks':6,'Lehigh':39,
               'Westmoreland':64,'Luzerne':40,
               'Northampton':48,'Dauphin':22,
               'Cumberland':21,'Erie':25,
               'Lackawanna':35,'Washington':62}

st.markdown("""
This tool provides interactive maps of crash events, either statewide or in one of the more populous counties.  Crash event dots are color-coded based on whether the crash involved serious cyclist injury, cyclist fatality, or neither.

Expand the menu below to adjust map options.
""")

with st.expander('Click here to expand or collapse map options menu'):
    geo = st.selectbox('Select either statewide or a particular county to plot:',
                       ['Statewide']+[county+' County' for county in county_data])
    animate = st.selectbox('Select how to animate the map:',
                           ['do not animate','by year','by month'])
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
    if county is not None:
        if animate_by == 'year':
            color_dots = len(crashes.query('COUNTY==@county[0] and CRASH_YEAR==2002')\
                       .BICYCLE_DEATH_COUNT.unique())+\
                        len(crashes.query('COUNTY==@county[0] and CRASH_YEAR==2002')\
                       .BICYCLE_SUSP_SERIOUS_INJ_COUNT.unique()) > 3
        else:
            color_dots = len(crashes.query('COUNTY==@county[0] and CRASH_YEAR==2002 and CRASH_MONTH==1')\
                       .BICYCLE_DEATH_COUNT.unique())+\
                       len(crashes.query('COUNTY==@county[0] and CRASH_YEAR==2002 and CRASH_MONTH==1')\
                       .BICYCLE_SUSP_SERIOUS_INJ_COUNT.unique()) > 3
if color_dots==False:
    st.markdown("""
    **Warning:** color-coding by injury/death status is disabled; this feature gives unexpected results
    when not all classes appear in the first animation frame due to bug/feature in Plotly animate functionality.
    Injury/death status is still visible in hover-text box.
    """)

fig = plot_map(df=crashes,county=county,animate=animate,color_dots=color_dots,
               animate_by=animate_by,show_fig=False,return_fig=True)

st.plotly_chart(fig,use_container_width=True)