## BikeSaferPA: understanding cyclist outcomes

This web app provides a suite of tools to accompany Eamonn Tweedy's [BikeSaferPA project](https://github.com/e-tweedy/BikeSaferPA). These tools allow the user to:
- Visualize data related to crashes involving bicycles in Pennsylvania during the years 2002-2021, which was collected from a publically available [PENNDOT crash dataset](https://pennshare.maps.arcgis.com/apps/webappviewer/index.html?id=8fdbf046e36e41649bbfd9d7dd7c7e7e).
- Experiment with the BikeSaferPA model, which was trained on this cyclist crash data and designed to predict severity outcomes for cyclists based on crash data.

### [Visit the web app](https://bike-safer-pa.streamlit.app/)

### Repository components:
- 'cyclists.csv' and 'crashes.csv' : datasets used for analysis
- 'app.py' : main streamlit app page
- 'study.pkl' : trained BikeSaferPA machine learning model
- 'app_data.pkl' : prepared data used for user input widget labels
- 'lib' : directory of custom modules
    - 'vis_data.py' : data visualization functions
    - 'transform_data'py' : data transformation functions
    - 'study_classif.py' : class for studying machine learning classifiers
