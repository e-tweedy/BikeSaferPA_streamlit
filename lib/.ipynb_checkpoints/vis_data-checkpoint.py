import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
	
def plot_map(df,city=None,county=None,animate=True,color_dots=True,animate_by='year',show_fig=True,return_fig=False):
    """
    Displays a plotly.express.scatter_mapbox interactive map
    of crashes in a municipality if specified, or otherwise
    statewide.  Can be animated over time or static.
    
    Parameters:
    -----------
    df : pd.DataFrame
        dataframe of crash samples
    city or county : tuple or None
        if provided, must be a tuple (code,name)
        - code : str
            the code corresponding to the desired municipality/county
            (see the data dictionary)
        - name : str
            the name you want to use for the municipality/county
            in plot title
        * At most one of these can be not None!
    animate : bool
        if animate==True, then the map will animate using
        the frequency provided in animate_by
    color_dots : bool
        if color_dots==True, then dots will be color-coded by
        'serious injury or death' status.
        WARNING: if color_dots and animate, then all frames
        will be missing samples in 'serious injury or death'
        classes which aren't present in first frame - due to
        bug in plotly animation_frame implementation.
        Recommend only using both when geographic
        area is statewide or at least has all values of
        'serious injury or death' in first frame
    animate_by : str
        the desired animation frequency, must be
        either 'year' or 'month'
    show_fig : bool
        whether to display figure using fig.show()
    return_fig : bool
        whether to return the figure object
   
   Returns: Either figure or None
   --------
    """
    assert (city is None)|(county is None), 'A city and county cannot both be provided.'
    # Copy df and create new column for color coding event type
    df = df.copy()
    df.loc[df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0,'Serious cyclist injury or death']='serious injury'
    df.loc[df.BICYCLE_DEATH_COUNT>0,'Serious cyclist injury or death']='death'
    df['Serious cyclist injury or death']=df['Serious cyclist injury or death'].fillna('neither')
    
    # Set animation parameters
    if animate:
        if animate_by == 'year':
            animation_frame = 'CRASH_YEAR'
            title_animate = ' by year'
        elif animate_by == 'month':
            df['DATE'] = pd.to_datetime((df['CRASH_MONTH'].astype('str')\
                                         +'-'+df['CRASH_YEAR'].astype('str')),
                                       format = "%m-%Y")
            df=df.sort_values(by='DATE')
            df['DATE']=df['DATE'].astype('str').apply(lambda x: x.rsplit('-',1)[0])
            animation_frame = 'DATE'
            title_animate = ' by month'
        else:
            raise ValueError("animate_by must be 'year' or 'month'")
    else:
        animation_frame = None
        title_animate = ''
    
    if color_dots:
        color='Serious cyclist injury or death'
    else:
        color=None
    
    # Adjustments for when city or county are provided
    if city is not None:
        df = df[df.MUNICIPALITY==city[0]]
        # Ignore extreme outlier samples - lat,lon may be incorrect
        df = df[np.abs(stats.zscore(df.DEC_LAT))<=4]
        df = df[np.abs(stats.zscore(df.DEC_LONG))<=4]
        title_place = city[1]+', PA'
    elif county is not None:
        df = df[df.COUNTY==county[0]]
        # Ignore extreme outlier samples - lat,lon may be incorrect
        df = df[np.abs(stats.zscore(df.DEC_LAT))<=4]
        df = df[np.abs(stats.zscore(df.DEC_LONG))<=4]
        title_place = county[1]+' county, PA'
    else:
        title_place = 'PA'
    
    # Compute default zoom level based on lat,lon ranges.
    # open-street-map uses 
    max_lat, min_lat = df.DEC_LAT.max(), df.DEC_LAT.min()
    max_lon, min_lon = df.DEC_LONG.max(), df.DEC_LONG.min()
    
    # 2^(zoom) = 360/(longitude width of 1 tile)
    zoom = np.log2(360/max(max_lon-min_lon,max_lat-min_lat))
    
    lat_center = (max_lat+min_lat)/2
    lon_center = (max_lon+min_lon)/2
    
    # Adjust width so that aspect ratio matches shape of state
    width_mult = (max_lon-min_lon)/(max_lat-min_lat)
    cols  = ['CRN','DEC_LAT','DEC_LONG','Serious cyclist injury or death','CRASH_YEAR','CRASH_MONTH']
    if animate_by=='month':
        cols.append('DATE')
    # Plot mapbox
    fig = px.scatter_mapbox(df, lat='DEC_LAT',lon='DEC_LONG',
                            color=color,
                            color_discrete_map={'neither':'royalblue','serious injury':'orange','death':'crimson'},
                            mapbox_style='open-street-map',
                            animation_frame = animation_frame,
                            animation_group='CRN',
                            hover_data = {'DEC_LAT':False,'DEC_LONG':False,
                                         'CRASH_YEAR':True,'CRASH_MONTH':True,
                                         'Serious cyclist injury or death':True},
                            width = width_mult*500,height=700,zoom=zoom,
                            center={'lat':lat_center,'lon':lon_center},
                            title=f'Crashes involving bicycles{title_animate}<br> in {title_place}, 2002-2021')
    fig.update_layout(legend=dict(orientation='h',xanchor='right',yanchor='bottom',x=1,y=-0.12),
                     legend_title_side='top')
    if show_fig:
        fig.show()
    if return_fig:
        return fig
	
def feat_perc(feat, df, col_name = 'percentage', feat_name = None):
    """
    Constructs a single-column dataframe 'perc'
    containing the value counts in the series
    df[feat] as percentages of the whole.
    - 'df' is the input dataframe.
    - 'feat' is the desired column of df.
    - 'col_name' is the name of the
    column of the output dataframe
    - 'feat_name' is the index name
    of the output dataframe if provided, otherwise
    will use 'feat' as index name.
    """
    perc = pd.DataFrame({col_name:df[feat].value_counts(normalize=True).sort_index()})
    if feat_name:
        perc.index.name=feat_name
    else:
        perc.index.name=feat
    return perc

def feat_perc_bar(feat,df,feat_name=None,cohort_name=None,show_fig=True,return_fig=False,sort=False):
    """
    Makes barplot of two series:
        - distribution of feature among all cyclists
        - distribution of feature among cyclists with serious injury or fatality

    Parameters:
    -----------
    feat : str
        The column name of the desired feature
    df : pd.DataFrame
        The input dataframe
    feat_name : str or None
        The feature name to use in the
        x-axis label.  If None, will use feat
    cohort_name : str or None
        qualifier to use in front of 'cyclists'
        in titles, if provided, e.g. 'rural cyclists'
    show_fig : bool
        whether to finish with fig.show()
    return_fig : bool
        whether to return the fig object
    sort : bool
        whether to sort bars. If False, will use default sorting
        by category name or feature value.  If True, will resort
        in descending order by percentage

    Returns: figure or None
    --------
    """
    if feat_name is None:
        feat_name=feat
    df_inj = df.query('SERIOUS_OR_FATALITY==1')
    table = feat_perc(feat,df)
    table.loc[:,'cohort']='all'
    ordering = list(table['percentage'].sort_values(ascending=False).index) if sort else None
    table_inj = feat_perc(feat,df_inj)
    table_inj.loc[:,'cohort']='seriously injured or killed'
    table = pd.concat([table,table_inj],axis=0).reset_index()
    category_orders = {'cohort':['all','seriously injured or killed']}
    if sort:
        category_orders[feat]=ordering
    fig = px.bar(table,y='cohort',x='percentage',color=feat,
                 barmode='stack',text_auto='.1%',
                category_orders=category_orders,
                title=f'Distributions of {feat} values within cyclist cohorts')
    fig.update_yaxes(tickangle=-90)
    fig.update_xaxes(tickformat=".0%")
    if show_fig:
        fig.show()
    if return_fig:
        return fig
    
# def feat_perc_comp(feat,df,feat_name=None,cohort_name = None,merge_inj_death=True):
#     """
#     Returns a styled dataframe (Styler object)
#     whose underlying dataframe has three columns
#     containing value counts of 'feat' among:
#     - all cyclists involved in crashes
#     - cyclists suffering serious injury or fatality
#     each formatted as percentages of the series sum.
#     Styled with bars comparing percentages

#     Parameters:
#     -----------
#     feat : str
#         The column name of the desired feature
#     df : pd.DataFrame
#         The input dataframe
#     feat_name : str or None
#         The feature name to use in the output dataframe
#         index name.  If None, will use feat
#     cohort_name : str or None
#         qualifier to use in front of 'cyclists'
#         in titles, if provided, e.g. 'rural cyclists'
#     merge_inj_death : bool
#         whether to merge seriously injured and killed cohorts
#     Returns:
#     --------
#     perc_comp : pd.Styler object
#     """
#     # Need qualifier for titles if restricting cyclist cohort
#     qualifier = cohort_name if cohort_name is not None else ''
    
#     # Two columns or three, depending on merge_inj_death
#     if merge_inj_death:
#         perc_comp = feat_perc(feat,df=df,feat_name=feat_name,
#                          col_name='all cyclists',)\
#                 .merge(feat_perc(feat,feat_name=feat_name,
#                                  df=df.query('SERIOUS_OR_FATALITY==1'),
#                                  col_name=qualifier+'cyclists with serious injury or fatality'),
#                       on=feat,how='left')
#         perc_comp = perc_comp[perc_comp.max(axis=1)>=0.005]
#     else:
#         perc_comp = feat_perc(feat,df=df,feat_name=feat_name,
#                          col_name='all cyclists')\
#                 .merge(feat_perc(feat,feat_name=feat_name,
#                                  df=df.query('INJ_SEVERITY=="susp_serious_injury"'),
#                                  col_name=qualifier+'cyclists with serious injury'),
#                       on=feat,how='left')\
#                 .merge(feat_perc(feat,feat_name=feat_name,
#                                  df=df.query('INJ_SEVERITY=="killed"'),
#                                  col_name=qualifier+'cyclists with fatality'),
#                       on=feat,how='left')
    
#     # If feature is not ordinal, sort rows descending by crash counts
#     if feat not in ['AGE_BINS','SPEED_LIMIT','DAY_OF_WEEK','HOUR_OF_DAY']:
#         perc_comp=perc_comp.sort_values(by='all cyclists',ascending=False)
    
#     # Relabel day numbers with strings
#     if feat == 'DAY_OF_WEEK':
#         perc_comp.index=['Sun','Mon','Tues','Wed','Thurs','Fri','Sat']
#         perc_comp.index.name='DAY_OF_WEEK'
#     perc_comp=perc_comp.fillna(0)
#     table_columns = list(perc_comp.columns)
    
#     # Define format for displaying floats
#     format_dict={col:'{:.2%}' for col in perc_comp.columns}

        
#     # Define table styles
#     styles = [dict(selector="caption",
#                    props=[("text-align", "center"),
#                           ("font-size", "100%"),
#                           ("color", 'black'),
#                           ("text-decoration","underline"),
#                           ("font-weight","bold")])]
    
#     # Return formatted dataframe
#     if feat_name is None:
#         feat_name=feat
#     caption = f'Breakdown of {feat_name} among cyclist groups'
#     return perc_comp.reset_index().style.set_table_attributes("style='display:inline'")\
#                                     .format(format_dict).bar(color='powderblue',
#                                     subset=table_columns).hide().set_caption(caption)\
#                                     .set_table_styles(styles)