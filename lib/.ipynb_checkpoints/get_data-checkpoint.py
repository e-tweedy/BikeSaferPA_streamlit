import pandas as pd

def extract_data(year):
    """
    A function for loading data corresponding to an individual
    year from a CSV file.  Data is then preprocessed and the
    following dataframes are returned:
    - 'bicycles': samples are bicycle vehicles which were
       involved in crashes.
    - 'persons': samples are all individuals involved in
      crashes involving bicycles.
    - 'crashes': samples are crash events involving bicycles.
    - 'roadway': additional features for
      crash events, related to roadway attributes and conditions.
    """
    
    # Retrieve vehicle samples corresponding to bicycles.
    # Note that in some samples VEH_TYPE is string, others float
    vehicles = pd.read_csv(f'data/raw_csv/VEHICLE_{year}_Statewide.csv',encoding='latin')
    bicycle_filter = vehicles.VEH_TYPE.isin([20,21,'20','21'])
    cols = ['CRN', 'GRADE', 'IMPACT_POINT',
            'RDWY_ALIGNMENT','UNIT_NUM',
            'VEH_MOVEMENT', 'VEH_POSITION','VEH_ROLE', 'VEH_TYPE']
    bicycles = vehicles[bicycle_filter][cols]
    del vehicles
    
    # Merge onto bicycles dataframe some additional features from cycle
    cycles = pd.read_csv(f'data/raw_csv/CYCLE_{year}_Statewide.csv',encoding='latin')
    cols = ['CRN','UNIT_NUM','PC_HDLGHT_IND', 'PC_HLMT_IND','PC_REAR_RFLTR_IND']
    bicycles = bicycles.merge(cycles[cols],how='left',on=['CRN','UNIT_NUM'])
    del cycles
    
    # Retrieve information about persons involved in crashes involving bikes
    # (not just the persons riding the bikes)
    persons = pd.read_csv(f'data/raw_csv/PERSON_{year}_Statewide.csv',encoding='latin')
    cols = ['AGE','CRN','INJ_SEVERITY','PERSON_TYPE',
            'RESTRAINT_HELMET','SEX', 'TRANSPORTED', 'UNIT_NUM']
    persons = persons[persons.CRN.isin(bicycles.CRN)][cols]
    
    # Retrieve crash samples involving bikes
    crashes = pd.read_csv(f'data/raw_csv/CRASH_{year}_Statewide.csv',encoding='latin')
    cols = ['CRN','ARRIVAL_TM','DISPATCH_TM','COUNTY','MUNICIPALITY','DEC_LAT','DEC_LONG',
            'BICYCLE_DEATH_COUNT','BICYCLE_SUSP_SERIOUS_INJ_COUNT',
            'BUS_COUNT','COMM_VEH_COUNT','HEAVY_TRUCK_COUNT','SMALL_TRUCK_COUNT','SUV_COUNT','VAN_COUNT',
            'CRASH_MONTH', 'CRASH_YEAR','DAY_OF_WEEK','HOUR_OF_DAY',
            'COLLISION_TYPE','ILLUMINATION','INTERSECT_TYPE',
            'LOCATION_TYPE','RELATION_TO_ROAD','TIME_OF_DAY',
            'ROAD_CONDITION','TCD_TYPE','TCD_FUNC_CD','URBAN_RURAL',
            'WEATHER1','WEATHER2']
    crashes = crashes[crashes.CRN.isin(bicycles.CRN)][cols]
    
    # Retrieve roadway data involving bikes
    roadway = pd.read_csv(f'data/raw_csv/ROADWAY_{year}_Statewide.csv',encoding='latin')
    cols = ['CRN','SPEED_LIMIT','RDWY_COUNTY']
    roadway = roadway[roadway.CRN.isin(bicycles.CRN)][cols]
    
    # Merge onto out bicycle_crashes and ped_crashes dataframe
    # some additional flag features.
    # Include flag features corresponding to driver impairment,
    # driver inattention, other driver attributes,relevant road conditions, etc.
    flags = pd.read_csv(f'data/raw_csv/FLAG_{year}_Statewide.csv',encoding='latin')
    cols = ['AGGRESSIVE_DRIVING','ALCOHOL_RELATED','ANGLE_CRASH','CELL_PHONE','COMM_VEHICLE',
            'CRN','CROSS_MEDIAN','CURVED_ROAD','CURVE_DVR_ERROR','DISTRACTED','DRINKING_DRIVER',
            'DRUGGED_DRIVER','DRUG_RELATED','FATIGUE_ASLEEP','HO_OPPDIR_SDSWP','ICY_ROAD',
            'ILLUMINATION_DARK','IMPAIRED_DRIVER','INTERSECTION','LANE_DEPARTURE',
            'NHTSA_AGG_DRIVING','NO_CLEARANCE',
            'NON_INTERSECTION','REAR_END','RUNNING_RED_LT','RUNNING_STOP_SIGN',
            'RURAL','SNOW_SLUSH_ROAD','SPEEDING','SPEEDING_RELATED',
            'SUDDEN_DEER','TAILGATING','URBAN','WET_ROAD','WORK_ZONE',
            'MATURE_DRIVER','YOUNG_DRIVER']
    crashes = crashes.merge(flags[cols],how='left',on='CRN')
    del flags
    
    return bicycles, persons, crashes, roadway