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

st.sidebar.title('BikeSaferPA')

from lib.vis_data import plot_map

@st.cache_data
def get_data(filename):
    return pd.read_csv(filename)

crashes = get_data('crashes.csv')

st.header('BikeSaferPA visualization suite')
st.subheader('Mapping bicycle crashes in PA')

city_data = {'Philadelphia':67301,'Pittsburgh':2301,
             'Allentown':39301,'Reading':6301,
             'Erie':25302,'Scranton':35302,
             'Harrisburg':22301,'State College':14410,
             'Upper Darby':23111,'Lower Merion':46104,
             'Bensalem':9202,'Abington':46101,
             'Bethlehem':39302,'Lancaster':36301}

st.markdown('This tool provides interactive maps showing crash events involving cyclists in PA, either statewide or in one of several population centers.  Choose a geographic locale and an animation method below.  Crash event dots are color-coded based on whether the crash involved serious cyclist injury, cyclist fatality, or neither.')

geo = st.selectbox('Select a geographic range for the plot:',
                   ['Statewide']+list(sorted(city_data.keys())))

if geo == 'Statewide':
    city = None
else:
    city = (city_data[geo],geo)
    
animate = st.selectbox('Select how to animate the map:',
                       ['do not animate','by year','by month'])

if animate == 'do not animate':
    animate = False
    animate_by=None
else:
    animate_by = animate.split(' ')[1]
    animate = True

fig = plot_map(df=crashes,city=city,animate=animate,
               animate_by=animate_by,show_fig=False,return_fig=True)

st.plotly_chart(fig,use_container_width=True)