import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import seaborn as sns
# from IPython.display import display, display_html
import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from plotly.offline import plot
from scipy import stats

# def plot_over_time(df = None, feature='CRASH_YEAR',label='year',kde=False,bw_adjust=[1,1,1],split_urban_rural = False,best_legend=False):
#     """
#     Displays figure containing three subplots, all
#     barplots of counts of the following over time:
#     1. All crashes involving bicycles from 2002-2021
#     2. Those which resulted in some serious injury of cyclist(s)
#     3. Those which resulted in fatality of cyclist(s)
    
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         dataframe of crash data
#     feature : str
#         feature to use on the x-axis (time frequency)
#         must be one of:
#         'CRASH_YEAR', 'CRASH_MONTH','DAY_OF_WEEK','TIME_OF_DAY'
#     label : str
#         the desired label for that feature to use in plot titles
#         recommended: 'year', 'month', 'day of the week', 'hour of the day'
#     kde : bool
#         if kind == 'hist' and kde is True, the plots will include
#         a kernel density estimate curve
#     bw_adjust : list
#         list of three positive floats, each will serve as the bandwidth
#         adjustment parameter for the corresponding plot's kde curve.
#         larger values provide smoother kde curves
#     split_urban_rural : bool
#         if split_urban_rural==True, then rural counts and urban/urbanized
#         counts will be separated into stacked barplot series
#     best_legend : bool
#         if best_legend==True, will set the legend position to 'best'.
#         otherwise, will set it to 'lower center'
        
#     Returns: None (figure displayed)
#     -------
#     """
    
#     df = df.copy()
#     plot_dict={}
    
#     # Additional parameters for stacked urban/rural bars
#     if split_urban_rural:
#         df['URBAN_RURAL'] = df['URBAN_RURAL'].where(df.URBAN_RURAL=='rural','urban or urbanized')
#         plot_dict['hue']='URBAN_RURAL'
#         plot_dict['hue_order']=['rural','urban or urbanized']
#         plot_dict['multiple']='stack'
    
#     # Define cohorts
#     df_inj = df[df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0]
#     df_death = df[df.BICYCLE_DEATH_COUNT>0]
#     df_inj_or_death = df[(df.BICYCLE_DEATH_COUNT>0)|(df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0)]
    
#     # Set axis label parameter values based on feature
#     if feature=='CRASH_YEAR':
#         span = range(2002,2022)
#         rot=45
#         tick_labels=span
#     elif feature=='CRASH_MONTH':
#         span = range(1,13)
#         rot=0
#         tick_labels = ['Jan','Feb','Mar','Apr','May','Jun',
#                        'Jul','Aug','Sep','Oct','Nov','Dec']
#     elif feature=='DAY_OF_WEEK':
#         span = range(1,8)
#         rot=0
#         tick_labels = ['Sun','Mon','Tues','Wed','Thurs','Fri','Sat']
#     elif feature=='HOUR_OF_DAY':
#         span = range(24)
#         rot=45
#         tick_labels = [str(x)+'a' for x in [12]+list(range(1,12))]\
#                      +[str(x)+'p' for x in [12]+list(range(1,12))]
#     else:
#         raise ValueError("feature must be one of 'CRASH_YEAR','CRASH_MONTH','MONTH_OF_YEAR','TIME_OF_DAY'")

#     # Initialize figure and plot on axes    
#     fig,axs = plt.subplots(1,3,figsize=(20,5))
#     fig.suptitle(f'Counts of crashes involving bicycles in PA by {label}, 2002-2021',fontsize='large')
    
#     sns.histplot(ax=axs[0],data=df, x = feature,bins=span[1]-span[0]+1,
#                  kde=kde,discrete=True,kde_kws={'bw_adjust':bw_adjust[0]},**plot_dict)
#     sns.histplot(ax=axs[1],data = df_inj,x=feature,
#                  bins=span[0]-span[0]+1,kde=kde,discrete=True,
#                  kde_kws={'bw_adjust':bw_adjust[1]},**plot_dict)
#     sns.histplot(ax=axs[2],data=df_death, x=feature,
#                  bins=span[0]-span[0]+1,kde=kde,discrete=True,
#                  kde_kws={'bw_adjust':bw_adjust[2]},**plot_dict)
    
#     if best_legend:
#         loc='best'
#     else:
#         loc='lower center'
    
#     # Format axes
#     for ax in axs:
#         ax.set_xticks(span,labels=tick_labels)
#             # ax.set_xticks(range(span[0],span[1]+1,(span[1]-span[0])//6))
#         ax.set_title(f'Incidence of {feature} values',fontsize='small')
#         ax.yaxis.set_tick_params(labelsize='small')
#         ax.xaxis.set_tick_params(labelsize='x-small',labelrotation=rot)
#         ax.set_ylabel(f'count of crashes per {label}',fontsize='small')
#         ax.set_xlabel(label,fontsize='small')
#         if split_urban_rural:
#             legend = ax.get_legend()
#             handles = legend.legendHandles
#             legend.remove()
#             ax.legend(handles, ['rural', 'urban or urbanized'], title='crash setting',loc=loc,ncol=2,
#                      columnspacing=1,fontsize='small',title_fontsize='small')
#     axs[0].set_title('All crashes' ,fontsize='medium')
#     axs[1].set_title('Crashes with serious cyclist injury',fontsize='medium')
#     axs[2].set_title('Crashes with cyclist fatality',fontsize='medium')
#     plt.tight_layout()
#     plt.show()
	
# def plot_month_series(df):
# 	"""
# 	Plots histplot of entire time series monthly crashes
# 	Parameters:
# 	-----------
# 	df : pd.DataFrame
# 		Should have columns 'CRASH_MONTH' and 'CRASH_YEAR'
# 		which both have type int
# 	Returns: None (plot displayed)
# 	-------
# 	"""
# 	# Create string feature DATE of the form 'MM-YYYY'
# 	df['DATE'] = pd.to_datetime((df['CRASH_MONTH'].astype('str')\
#                                         +'-'+df['CRASH_YEAR'].astype('str')),
#                                       format = "%m-%Y")
# 	df=df.sort_values(by='DATE')
# 	df['DATE']=df['DATE'].astype('str').apply(lambda x: x.rsplit('-',1)[0])
	
# 	fig,axs=plt.subplots(3,1,figsize=(20,7),sharex=True)
# 	fig.suptitle('Monthly counts of df involving bicycles in PA, 2002-2021',fontsize='large')
# 	sns.histplot(ax=axs[0],data = df, x='DATE')
# 	sns.histplot(ax=axs[1],data = df[df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0],x='DATE')
# 	sns.histplot(ax=axs[2],data = df[df.BICYCLE_DEATH_COUNT>0],x='DATE')

# 	# Format axes
# 	for ax in axs:
# 		ax.yaxis.set_tick_params(labelsize='small')
# 		ax.set_ylabel(f'count',fontsize='small')
# 		ax.set_xlabel(None)
# 	axs[2].xaxis.set_tick_params(labelsize='x-small',labelrotation=45,labelleft=True)
# 	axs[2].set_xticks(range(0,240,12),labels=df.CRASH_YEAR.unique())
# 	axs[0].set_title('All crashes',fontsize='medium')
# 	axs[1].set_title('Crashes with serious cyclist injury',fontsize='medium')
# 	axs[2].set_title('Crashes with cyclist fatality',fontsize='medium')
# 	plt.tight_layout()
# 	plt.show()

# def compute_perc_change(df,period='year',title='all crashes'):
#     """
#     Computes period-over-period percent changes
#     of the per-period count of samples
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         dataframe of crash data
#     period : str
#         period to use - must be 'year' or 'month'
#     title : str
#         title for the column
        
#     Returns:
#     --------
#     perc_change : pd.DataFrame
#         dataframe showing period-over-period
#         percent changes in sample count
#         - one columns corresponding to the
#           sample set included in df
#         - rows correspond to time steps
#     """
#     # Calculate series of percent changes between timesteps
#     if period == 'year':
#         feat='CRASH_YEAR'
#         first, last = df[feat].min(), df[feat].max()
#         drop_time_steps=[first-1,first]
#     elif period == 'month':
#         feat='CRASH_MONTH'
#         first, last = df[feat].min(), df[feat].max()
#         drop_time_steps=first-1
#     else:
#         raise ValueError("period must be 'year' or 'month'")
    
#     counts = df[feat].value_counts()
#     counts.loc[first-1]=counts[last]
#     counts=counts.sort_index()
#     perc_change = np.round(100*(counts-counts.shift(1))/counts.shift(1),1).drop(drop_time_steps,axis=0)
    
#     # Convert to dataframe
#     perc_change = pd.DataFrame({title:perc_change})
#     perc_change.index.name=period
#     return perc_change

# def perc_change_table(df,period='year'):
#     """
#     Display a styled dataframe of period-over-period
#     percent changes in sample count among various
#     crash sample sets
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         dataframe of crash data
#     period : str
#         period to use - must be 'year' or 'month'
    
#     Returns:
#     --------
#     None; displays a styled dataframe:
#         - rows correspond to four subsets among crash samples:
#             - all crashes involving bicycles
#             - crashes with serious cyclist injury
#             - crashes with cyclist fatality
#             - crashes with either serious cyclist injury
#               or cyclist fatality
#         - columns correspond to time steps
#         - each cell contains the percent change of the
#           count of the corresponding subset from the
#           previous time step to the current time step
#         - the cell is shaded based on the percent change
#             - blue for decrease, red for increase
#             - intensity of color indicates 
#               magnitude of percent change
#     """
#     # Define cohorts
#     df_inj = df[df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0]
#     df_death = df[df.BICYCLE_DEATH_COUNT>0]
#     df_inj_or_death = df[(df.BICYCLE_DEATH_COUNT>0)|(df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0)]
    
#     # Build dataframe by merging calls to compute_perc_change
#     perc_change = compute_perc_change(df,period=period).merge(compute_perc_change(df_inj,period=period,
#                                                                          title='with serious injury'),on=period)\
#                                           .merge(compute_perc_change(df_death,period=period,
#                                                                      title='with fatality'),on=period)\
#                                          .merge(compute_perc_change(df_inj_or_death,period=period,
#                                                                     title='with either'),on=period)\
#                                          .rename_axis(f'{period}ly change of:',axis='columns').transpose()
#     # Display stylized dataframe
#     display((perc_change.astype('str')+'%').style.background_gradient(axis=None,cmap='bwr',gmap=perc_change,vmin=-100,vmax=100))
	
def plot_map(df,city=None,animate=True,animate_by='year',show_fig=True,return_fig=False):
    """
    Displays a plotly.express.scatter_mapbox interactive map
    of crashes in a municipality if specified, or otherwise
    statewide.  Can be animated over time or static.
    
    Parameters:
    -----------
    df : pd.DataFrame
        dataframe of crash samples
    city : tuple or none
        if provided, must be a tuple (code,name)
        - code : str
            the code corresponding to the desired municipality
            (see the data dictionary)
        - name : str
            the name you want to use for the municipality
            in plot title
    animate : bool
        if animate==True, then the map will animate using
        the frequency provided in animate_by
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
    # Copy df and create new column for color coding event type
    df = df.copy()
    df.loc[df.BICYCLE_SUSP_SERIOUS_INJ_COUNT>0,'Serious cyclist injury or death']='death'
    df.loc[df.BICYCLE_DEATH_COUNT>0,'Serious cyclist injury or death']='serious injury'
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
    
    # Adjustments for when city is provided 
    if city is not None:
        df = df[df.MUNICIPALITY==city[0]]
        # Ignore extreme outlier samples - lat,lon may be incorrect
        df = df[np.abs(stats.zscore(df.DEC_LAT))<=4]
        df = df[np.abs(stats.zscore(df.DEC_LONG))<=4]
        title_place = city[1]+', PA'
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
    
    # Plot mapbox
    fig = px.scatter_mapbox(df, lat='DEC_LAT',lon='DEC_LONG',
                            color='Serious cyclist injury or death',
                            color_discrete_map={'neither':'royalblue','serious injury':'orange','death':'crimson'},
                            mapbox_style='open-street-map',
                            animation_frame = animation_frame,
                            hover_data = {'DEC_LAT':False,'DEC_LONG':False,
                                         'CRASH_YEAR':True,'CRASH_MONTH':True},
                            width = width_mult*500,height=700,zoom=zoom,
                            center={'lat':lat_center,'lon':lon_center},
                            title=f'Crashes involving bicycles{title_animate} in {title_place}, 2002-2021')
    fig.update_layout(legend=dict(orientation='h',xanchor='right',yanchor='bottom',x=1,y=1.02))
    if show_fig:
        fig.show()
    if return_fig:
        return fig
	
# def feat_perc(feat, df, col_name = 'percentage', feat_name = None):
#     """
#     Constructs a single-column dataframe 'perc'
#     containing the value counts in the series
#     df[feat] as percentages of the whole.
#     - 'df' is the input dataframe.
#     - 'feat' is the desired column of df.
#     - 'col_name' is the name of the
#     column of the output dataframe
#     - 'feat_name' is the index name
#     of the output dataframe if provided, otherwise
#     will use 'feat' as index name.
#     """
#     perc = pd.DataFrame({col_name:df[feat].value_counts(normalize=True).sort_index()})
#     if feat_name:
#         perc.index.name=feat_name
#     else:
#         perc.index.name=feat
#     return perc
    
# def feat_perc_comp(feat,df,feat_name=None,cohort_name = None,merge_inj_death=True):
#     """
#     Returnes a styled dataframe (Styler object) 'perc_comp'
#     whose underlying dataframe has three columns
#     containing value counts of 'feat' among:
#     - all cyclists involved in crashes
#     - cyclists suffering serious injury or fatality
#     each formatted as percentages of the series sum.
#     Styled with bars comparing percentages
#     Inputs:
#     - 'df' is cyclists by default, but can be changed
#     e.g. if you want to filter the samples
#     - 'feat' if the desired feature to use
#     - 'feat_name' is the index name
#     of the output dataframe if provided, otherwise
#     will use 'feat' as index name.
#     - 'cohort_name' is a qualifier in front
#     of "cyclists" in titles, if provided
#     (e.g. "urban")
#     - if merge_inj_death is set to False,
#     then cyclists suffering serious injury and
#     the cyclists suffering fatality will be separated
#     into separate columns (three total columns)
    
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
#                                     subset=table_columns).hide_index().set_caption(caption)\
#                                     .set_table_styles(styles)

# def gray_empty(val):
#     """
#     Function for styling a pd.DataFrame,
#     where a cell will be given a gray background
#     if it contains '' otherwise no background color
    
#     Parameters:
#     -----------
#     val : any
#         the entry of a cell in the dataframe
#     Returns:
#     --------
#     'background-color: %s' % color : str
#         a CSS style tag, where color is 'gray' if val == '',
#         else color is ''
#     """
#     color = 'gray' if val=='' else ''
#     return 'background-color: %s' % color

# def highlight_diag(df):
#     """
#     Function for styling a pd.DataFrame,
#     where a cell will be given a gold background
#     if it lies on the main diagonal, otherwise
#     no background color
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         the given dataframe
    
#     Returns:
#     --------
#     diag_mask : pd.DataFrame
#         a dataframe with the same shape, indices, and columns as df
#         which contains CSS style tags as its entries.  entries on
#         the main diagonal are 'background-color: gold' and others are ''
#     """
#     diag_mask = pd.DataFrame("", index=df.index, columns=df.columns)
#     min_axis = min(diag_mask.shape)
#     diag_mask.iloc[range(min_axis), range(min_axis)] = 'background-color: gold'
#     return diag_mask

# def crosstab_percent(filters,df,target='SERIOUS_OR_FATALITY',thresh=0.001):
#     """
#     Generates two dataframes based on the count of samples with target==1
#     for each pair of filters in a list provided.
    
#     Parameters:
#     ----------
#     df : pd.DataFrame
#         the input dataframe
#     filters : dict
#         key:value pairs are filter_name:filter where
#             - filter : pd.DataFrame
#                 a mask for df, i.e. a dataframe with the same
#                 shape, indices, and columns of df whose entries
#                 are bools.
#             - filter_name : str
#                 the name of the filter, to be used in titles
#     target : str
#         the name of the target feature
#     thresh : float
#         the desired cutoff threshold for displaying values
    
#     Returns:
#     --------
#     crosstabs : pd.DataFrame
#         A symmetric dataframe:
#             - indices correspond to filters (names from filter_name)
#             - columns correspond to filters (names from filter_name)
#             - each entry is the count of samples in df which pass both filters
#               corresponding to that entry's row and column, as a string
    
#     percents : pd.DataFrame
#         A symmetric dataframe:
#             - indices correspond to filters (titles from filter_name)
#             - columns correspond to filters (titles from filter_name)
#             - each entry is the percent of samples which have target==1,
#               among those samples in df which pass both filters
#               corresponding to that entry's row and column,
#               rounded to one decimal place, as a string

#     Note that if an entry in crosstabs is smaller than
#     thresh*(number of samples in df), then the corresponding entry
#     of crosstabs and percents is replaced with the
#     empty string.
#     """
#     # Count of seriously injured or killed per filter pair
#     crosstabs = np.array([[df[(filters[f1])&(filters[f2])].shape[0]\
#                         for f1 in filters]\
#                        for f2 in filters ])
#     crosstabs = pd.DataFrame(crosstabs,index=filters.keys(),columns=filters.keys())
    
#     # Percent seriouslty injured or killed per filter pair
#     percents = np.array([[round(100*df[(filters[f1])&(filters[f2])]\
#                                 [target].sum()/df[(filters[f1])&(filters[f2])].shape[0],2)\
#                         for f1 in filters]\
#                        for f2 in filters ])
#     percents = pd.DataFrame(percents,index=filters.keys(),columns=filters.keys())
    
#     # Return dataframes with string type entries
#     percents = percents[crosstabs>=thresh*df.shape[0]].replace(np.nan,'').astype('str')
#     crosstabs = crosstabs[crosstabs>=thresh*df.shape[0]].replace(np.nan,'').astype('str')
#     return crosstabs, percents

# def stylize_dataframe(df):
#     """
#     Displays a stylized version of the dataframe df:
#     - Cells containing '' np.nan are colored gray
#     - Cells containing neither '' nor np.nan which lie
#       on the main diagonal are colored yellow
#     - Cells containing neither '' nor np.nan which lie
#       off of the main diagonal use a gradient color map:
#         - Cell (i,j) is colored red if its entry is greater than
#           the entry in cell (j,j), with shade indicating difference
#         - Cell (i,j) is colored blue if its entry is less than
#           the entry in cell (j,j), with shade indicating difference
          
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         input dataframe whose entries are either np.nan, '',
#         or strings which can be interpreted as floats
    
#     Returns: None (displays stylized dataframe)
#     --------
#     """
#     # change strings to float and np.nan before defining gradient map
#     df=df.replace('',np.nan).astype('float')
#     gmap = pd.DataFrame(0,index=df.index,columns=df.columns)
#     for col in df.columns:
#         gmap[col] = df[col]-df.loc[col,col]
#         gmap[col] = gmap[col].where(gmap[col]>=0,5*gmap[col])
#     # display styled dataframe
#     display((df.astype('str')+'%').replace('nan%','').style\
#             .background_gradient(axis=None, gmap=gmap, cmap='bwr',vmin=-gmap.max().max())\
#             .apply(highlight_diag,axis=None).applymap(gray_empty))