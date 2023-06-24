import pandas as pd
import numpy as np
import streamlit as st
import calendar as cal

st.sidebar.title('BikeSaferPA')
st.header('BikeSaferPA visualization suite')

st.markdown(
"""
This web app provides a suite of tools to accompany Eamonn Tweedy's [BikeSaferPA project](https://github.com/e-tweedy/BikeSaferPA), which allow the user to:
- Visualize data related to crashes involving bicycles in Pennsylvania during the years 2002-2021.
- Experiment with the BikeSaferPA model, which was trained on this dataset and designed to predict severity outcomes for cyclists based on crash data.

The tools and the BikeSaferPA model make use of the publically available [PENNDOT crash dataset](https://pennshare.maps.arcgis.com/apps/webappviewer/index.html?id=8fdbf046e36e41649bbfd9d7dd7c7e7e).  Use the navigation menu to the left to try them out!
"""
)
