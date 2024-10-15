import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering

from datetime import datetime, time
import json

import logging
import logstash
import sys
 
host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class CalcDist:
    ''' Implementing Distance Calculation from 2 points '''

    @classmethod
    def haversine_np(cls, lon1, lat1, lon2, lat2):
        try:
            km_data = pd.DataFrame(columns=['fac_latitude', 'fac_longitude','distance'])
            km_data['fac_latitude'] = lat2
            km_data['fac_longitude'] = lon2
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6371000 * c    
            km_data['distance'] = km
            return km_data
        except Exception as e:
            logging.error('Failed to execute function haversine_np : '+ str(e) )
            test_logger.error('Failed to execute function haversine_np : '+ str(e))
            return {"message": "internal server error, haversine_np", "code": 501}, 501

    @classmethod
    def distance_data(cls,stopss_copy,data_set_hundred_filtered_loop):
        try:
            stop_lon = stopss_copy['fac_longitude'].iloc[0]
            stop_lat = stopss_copy['fac_latitude'].iloc[0]
            df = pd.DataFrame(data={'lon1':data_set_hundred_filtered_loop['longitude'],'lon2':stop_lon,'lat1':data_set_hundred_filtered_loop['latitude'],'lat2':stop_lat})
            closest_stops = CalcDist.haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2'])
            closest_stops['closest_stop_geom'] = [','.join(str(x) for x in y) for y in map(tuple, closest_stops[['fac_latitude', 'fac_longitude']].values)]
            closest_stops['KM_distance'] = closest_stops['distance'].apply(lambda x: x/1000)
            data_set_copy1 = data_set_hundred_filtered_loop.join(closest_stops)
            return data_set_copy1
        except Exception as e:
            logging.error('Failed to execute function distance_data : '+ str(e) )
            test_logger.error('Failed to execute function distance_data : '+ str(e))
            return {"message": "internal server error, distance_data", "code": 501}, 501

    @classmethod
    def distance_data_facility(cls,stopss_copy,data_set_hundred_filtered_loop):
        try:
            stop_lon = stopss_copy['fac_longitude'].iloc[0]
            stop_lat = stopss_copy['fac_latitude'].iloc[0]
            df = pd.DataFrame(data={'lon1':data_set_hundred_filtered_loop['longitude'],'lon2':stop_lon,'lat1':data_set_hundred_filtered_loop['latitude'],'lat2':stop_lat})
            closest_stops = CalcDist.haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2'])
            closest_stops['closest_stop_geom'] = [','.join(str(x) for x in y) for y in map(tuple, closest_stops[['fac_latitude', 'fac_longitude']].values)]
            closest_stops['DistanceToOffice'] = closest_stops['distance'].apply(lambda x: x/1000)
            data_set_copy1 = data_set_hundred_filtered_loop.join(closest_stops)
            return data_set_copy1
        except Exception as e:
            logging.error('Failed to execute function distance_two_points: '+ str(e) )
            test_logger.error('Failed to execute function distance_two_points : '+ str(e))
            return {"message": "internal server error, distance_data", "code": 501}, 501

    @classmethod
    def create_stop_df(cls, stop_lat,stop_lon):
        try:
            stop_data = pd.DataFrame(columns=['fac_latitude', 'fac_longitude'])
            stop_data.at[0,'fac_latitude'] = stop_lat
            stop_data.at[0,'fac_longitude'] = stop_lon
            return stop_data
        except Exception as e:
            logging.error('Failed to execute function create_stop_df: '+ str(e) )
            test_logger.error('Failed to execute function create_stop_df : '+ str(e))
            return {"message": "internal server error create_stop_df", "code": 501}, 501

    #finding maxdistance, stopsdata, index
    @classmethod
    def find_max(cls,filtered_data):
        try:
            one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()]
            maxlat = one_data.latitude.iloc[0]
            maxlon = one_data.longitude.iloc[0]
            stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
            return one_data,stopss_copy
        except Exception as e:
            logging.error('Failed to execute function find_max: '+ str(e) )
            test_logger.error('Failed to execute function find_max : '+ str(e))
            return {"message": "internal server error find_max", "code": 501}, 501

    @classmethod
    def find_min(cls,filtered_data):
        try:
            one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()]
            maxlat = one_data.latitude.iloc[0]
            maxlon = one_data.longitude.iloc[0]
            stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
            return one_data,stopss_copy
        except Exception as e:
            logging.error('Failed to execute function find_min: '+ str(e) )
            test_logger.error('Failed to execute function find_min : '+ str(e))
            return {"message": "internal server error find_min", "code": 501}, 501


    @classmethod
    def silhouette(cls, cleaned_records):
        try:
            silhouette_scores = [] 
            X=cleaned_records.loc[:,['latitude','longitude']]
            for n_cluster in range(2, 6):
                silhouette_scores.append(silhouette_score(X, KMeans(n_clusters = n_cluster).fit_predict(X))) 
                
            max_ = silhouette_scores[0]
            
            max_index = 0
            for index, value in enumerate(silhouette_scores):
                if value > max_:
                    max_ = value
                    max_index = index+2

            if max_index == 0:
                max_index = 2

            return max_index,X
        except Exception as e:
            logging.error('Failed to execute function silhouette: '+ str(e) )
            test_logger.error('Failed to execute function silhouette : '+ str(e))
            return {"message": "internal server error, function silhouette failed to execute", "code": 501}, 501


    @classmethod
    def silhouette_below_ten(cls, cleaned_records, limit_size):
        try:
            silhouette_scores = [] 
            X=cleaned_records.loc[:,['latitude','longitude']]
            for n_cluster in range(2, limit_size):
                silhouette_scores.append(silhouette_score(X, KMeans(n_clusters = n_cluster).fit_predict(X))) 
                    
            max_ = silhouette_scores[0]
            
            max_index = 0
            for index, value in enumerate(silhouette_scores):
                if value > max_:
                    max_ = value
                    max_index = index+2

            if max_index == 0:
                max_index = 2

            return max_index,X
        except Exception as e:
            logging.error('Failed to execute function silhouette_below_ten: '+ str(e) )
            test_logger.error('Failed to execute function silhouette_below_ten : '+ str(e))
            return {"message": "internal server error, function silhouette_below_ten failed to execute", "code": 501}, 501

    @classmethod
    def add_to_dataframe_lesser(cls,filtered_data,less_than_radiuss,facility_stop_data,linestring_df2,occur_val):
        try:
            filtered_data['Route'] = occur_val
            linestring_df2 = linestring_df2.append(filtered_data, ignore_index=True)
            return linestring_df2
        except Exception as e:
            logging.error('Failed to execute function add_to_dataframe_lesser: '+ str(e) )
            test_logger.error('Failed to execute function add_to_dataframe_lesser : '+ str(e))
            return {"message": "internal server error, add_to_dataframe", "code": 501}, 501
    



    @classmethod
    def split_dataframe(cls,df, chunk_size = 10000): 
        try:
            chunks = list()
            num_chunks = len(df) // chunk_size + 1
            for i in range(num_chunks):
                chunks.append(df[i*chunk_size:(i+1)*chunk_size])
            return chunks    
        except Exception as e:
            logging.error('Failed to execute function split_dataframe: '+ str(e) )
            test_logger.error('Failed to execute function split_dataframe : '+ str(e))
            return {"message": "internal server error, split_dataframe", "code": 501}, 501


    @classmethod
    def distance_two_points(cls,lon1, lat1, lon2, lat2):
        try:
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6371000 * c 
            km = km/1000
            return km 
        except Exception as e:
            logging.error('Failed to execute function distance_two_points: '+ str(e) )
            test_logger.error('Failed to execute function distance_two_points : '+ str(e))
            return {"message": "internal server error, distance_two_pointscls", "code": 501}, 501

    @classmethod
    def Drop_Index_and_Reset(cls,LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon):
        try:
            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
            #add distance between
            first_val_status = True
            for index,row in LS_TN_Radius_Original.iterrows():
                if first_val_status:
                    ssslon = float(ssslon)
                    ssslat = float(ssslat)
                    oldlat = row.latitude
                    oldlon = row.longitude
                    dist_val = 0
                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                    LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                    first_val_status = False
                else:
                    newlat = row.latitude
                    newlon = row.longitude
                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                    LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                    oldlat = newlat
                    oldlon = newlon
        
            LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].astype(float)
            LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].round(3)

            timebtw = 0
            for index,row in LS_TN_Radius_Original.iterrows():
                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                LS_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
            
            timetooffice = 0
            for index,row in LS_TN_Radius_Original.iterrows():
                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                LS_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
            
            return LS_TN_Radius_Original
        except Exception as e:
            logging.error('Failed to execute function Drop_Index_and_Reset: '+ str(e) )
            test_logger.error('Failed to execute function Drop_Index_and_Reset : '+ str(e))
            return {"message": "internal server error, Drop_Index_and_Reset", "code": 501}, 501

    @classmethod
    def isNowInTimePeriod(cls,startTime, endTime, nowTime):
        try:
            startTime = datetime.strptime(startTime, "%I:%M%p")
            endTime = datetime.strptime(endTime, "%I:%M%p")
            nowTime = datetime.strptime(nowTime, "%I:%M%p")
            if startTime < endTime:
                return nowTime >= startTime and nowTime <= endTime
            else: #Over midnight
                return nowTime >= startTime or nowTime <= endTime
        except Exception as e:
            logging.error('Failed to execute function isNowInTimePeriod: '+ str(e) )
            test_logger.error('Failed to execute function isNowInTimePeriod : '+ str(e))
            return {"message": "internal server error isNowInTimePeriod", "code": 501}, 501

    @classmethod
    def DistanceBTWEMP_Calculate(cls,generated_df,ssslon,ssslat):
        try:
            first_val_status = True
            for index,row in generated_df.iterrows():
                if first_val_status:
                    ssslon = float(ssslon)
                    ssslat = float(ssslat)
                    oldlat = row.latitude
                    oldlon = row.longitude
                    dist_val = 0
                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                    generated_df.at[index,'DistanceBTWEMP'] = dist_val
                    first_val_status = False
                else:
                    newlat = row.latitude
                    newlon = row.longitude
                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                    generated_df.at[index,'DistanceBTWEMP'] = dist_val
                    oldlat = newlat
                    oldlon = newlon
                
            generated_df['DistanceBTWEMP'] = generated_df['DistanceBTWEMP'].astype(float)
            generated_df['DistanceBTWEMP'] = generated_df['DistanceBTWEMP'].round(3)
            return generated_df
        except Exception as e:
            logging.error('Failed to execute function DistanceBTWEMP_Calculate: '+ str(e) )
            test_logger.error('Failed to execute function DistanceBTWEMP_Calculate : '+ str(e))
            return {"message": "internal server error, DistanceBTWEMP_Calculate", "code": 501}, 501

    @classmethod
    def drop_minutes_norule(cls,filter_data_limit_seats,hour_max_set,ssslat,ssslon): 
        try:
            #do for minutes and No Rule
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    #total minutes taken
                    #Regenerate First Position Proper Distance
                    first_val_status = True
                    for index,row in single_dataset.iterrows():
                        if first_val_status:
                            ssslon = float(ssslon)
                            ssslat = float(ssslat)
                            oldlat = row.latitude
                            oldlon = row.longitude
                            dist_val = 0
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            first_val_status = False
                        else:
                            newlat = row.latitude
                            newlon = row.longitude
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            oldlat = newlat
                            oldlon = newlon
                        
        
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].astype(float)
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].round(3)
                    
                    timebtw = 0
                    for index,row in single_dataset.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        single_dataset.at[index,'TimeBtwEmp'] = timebtw
                    
                    timetooffice = 0
                    for index,row in single_dataset.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        single_dataset.at[index,'TimetoOffice'] = timetooffice 
                
                    TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['TimeBtwEmp']
                    if TotalKMCovered < hour_max_set:
                        single_object_df['Total_Person'] = len(single_dataset)
                        single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                        single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                        json_data = single_dataset.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                    else:
                        Total_minutes_covered = 0
                        list_index_val = []
                        for index,row in single_dataset.iterrows(): 
                            Total_minutes_covered = Total_minutes_covered+row.TimeBtwEmp
                            if Total_minutes_covered <= hour_max_set:
                                list_index_val.append(index)
                        
                        if len(list_index_val):
                            single_set_filtered  = single_dataset[single_dataset.index.isin(list_index_val)]
                        else:
                            single_set_filtered = single_dataset.head(1)
                        
        
                        TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                        single_object_df['Total_Person'] = len(single_set_filtered)
                        single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                        single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                        json_data = single_set_filtered.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
            
            Total_Emp_Sets_All = []
            for i,x in enumerate(all_set_emp):
                singleset_all = []
                singleset_all.append(i)
                singleset_all.append(x[0]['Total_Person'])
                singleset_all.append(x[0]['Total_Minutes_Covered'])
                Total_Emp_Sets_All.append(singleset_all)
            
            #find the maximum number count from the dict
            max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
            #filter only max_employee count  
            max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
            #filter only minimum data
            max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
            #filter only single list with covers more employee and less km 
            max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
            #filter only perticular set and return it.
            dataaset = all_set_emp[max_people_list[0][0]]
            return dataaset    
        except Exception as e:
            logging.error('Failed to execute function drop_minutes_norule: '+ str(e) )
            test_logger.error('Failed to execute function drop_minutes_norule : '+ str(e))
            return {"message": "internal server error, drop_minutes_norule", "code": 501}, 501


    @classmethod
    def Generate_Trip_Hour_NoRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,hour_max_set,ssslat,ssslon): 
        try:
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID 
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                else:
                    break
            
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val
                
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID   
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID  
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                else:
                    break
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_Hour_NoRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_Hour_NoRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_Hour_NoRule", "code": 501}, 501

    
    @classmethod
    def drop_minutes_norule_KM(cls,filter_data_limit_seats,distance_max_set,ssslat,ssslon): 
        try:
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    #total minutes taken
                    #Regenerate First Position Proper Distance
                    first_val_status = True
                    for index,row in single_dataset.iterrows():
                        if first_val_status:
                            ssslon = float(ssslon)
                            ssslat = float(ssslat)
                            oldlat = row.latitude
                            oldlon = row.longitude
                            dist_val = 0
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            first_val_status = False
                        else:
                            newlat = row.latitude
                            newlon = row.longitude
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            oldlat = newlat
                            oldlon = newlon
                        
        
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].astype(float)
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].round(3)
                    
                    timebtw = 0
                    for index,row in single_dataset.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        single_dataset.at[index,'TimeBtwEmp'] = timebtw
                    
                    timetooffice = 0
                    for index,row in single_dataset.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        single_dataset.at[index,'TimetoOffice'] = timetooffice 
                
                    TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                    if TotalKMCovered < distance_max_set:
                        single_object_df['Total_Person'] = len(single_dataset)
                        single_object_df['Total_KM_Covered'] = TotalKMCovered
                        single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                        json_data = single_dataset.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                    else:
                        Total_KM_covered = 0
                        list_index_val = []
                        for index,row in single_dataset.iterrows(): 
                            Total_KM_covered = Total_KM_covered+row.DistanceBTWEMP
                            if Total_KM_covered <= distance_max_set:
                                list_index_val.append(index)
                        
                        if len(list_index_val):
                            single_set_filtered  = single_dataset[single_dataset.index.isin(list_index_val)]
                        else:
                            single_set_filtered = single_dataset.head(1)
                        
                        TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                        single_object_df['Total_Person'] = len(single_set_filtered)
                        single_object_df['Total_KM_Covered'] = TotalKMCovered
                        single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                        json_data = single_set_filtered.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
            
            Total_Emp_Sets_All = []
            for i,x in enumerate(all_set_emp):
                singleset_all = []
                singleset_all.append(i)
                singleset_all.append(x[0]['Total_Person'])
                singleset_all.append(x[0]['Total_KM_Covered'])
                Total_Emp_Sets_All.append(singleset_all)
            
            #find the maximum number count from the dict
            max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
            #filter only max_employee count  
            max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
            #filter only minimum data
            max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
            #filter only single list with covers more employee and less km 
            max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
            #filter only perticular set and return it.
            dataaset = all_set_emp[max_people_list[0][0]]
            return dataaset 
        except Exception as e:
            logging.error('Failed to execute function drop_minutes_norule_KM: '+ str(e) )
            test_logger.error('Failed to execute function drop_minutes_norule_KM : '+ str(e))
            return {"message": "internal server error, drop_minutes_norule_KM", "code": 501}, 501


    @classmethod
    def Generate_Trip_no_validation_NoRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,ssslat,ssslon): 
        try:
            #start the code here
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        index_values_filtered = filter_data_limit_seats.index.values
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        
                        first_val_status = True
                        for index,row in LS_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                    
                    else:
                        
                        filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        index_values_filtered = filter_data_limit_seats.index.values
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        
                        first_val_status = True
                        for index,row in LS_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
            
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                else:
                    break
                
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val  
            
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        index_values_filtered = filter_data_limit_seats.index.values
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        
                        first_val_status = True
                        for index,row in GR_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                    
                    else:
                        
                        filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        index_values_filtered = filter_data_limit_seats.index.values
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        
                        first_val_status = True
                        for index,row in GR_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
            
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                else:
                    break

            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_no_validation_NoRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_no_validation_NoRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_no_validation_NoRule", "code": 501}, 501

    @classmethod
    def Generate_Trip_KM_NoRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,distance_max_set,ssslat,ssslon): 
        try:
            #start the code here
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule_KM(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)         
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule_KM(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID 
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                else:
                    break

            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val

            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule_KM(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)         
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_norule_KM(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID 
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)            
                else:
                    break

            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_KM_NoRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_KM_NoRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_KM_NoRule", "code": 501}, 501

    @classmethod
    def Generate_Trip_no_validation_WithRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,ssslat,ssslon,AssignEscortStatus): 
        try:
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                        #get the Last employee Details
                        last_emp = filter_data_limit_seats.tail(1)
                        Gender_Type = last_emp.Gender.iloc[0]
                        Last_emp_index = last_emp.index.values
                        if Gender_Type == 'Female':
                            filter_data_limit_seats.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                #drop the data from main
                                index_values_filtered = filter_data_limit_seats.index.values
                                LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        else:
                            index_values_filtered = filter_data_limit_seats.index.values
                            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            trip_assign_val = trip_assign_val + 1
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                    
                    
                        #reset to actual distance
                        first_val_status = True
                        for index,row in LS_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in LS_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            LS_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                    
                    else:
                        
                        if len(LS_TN_Radius_Original) == 1:
                            filter_data_limit_seats = LS_TN_Radius_Original.head(1)
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                index_values_filtered = filter_data_limit_seats.index.values
                                LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                    
                        else:
                            filter_data_limit_seats = LS_TN_Radius_Original.head(len(LS_TN_Radius_Original))
                            
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                filter_data_limit_seats.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                last_emp = filter_data_limit_seats.tail(1)
                                Gender_Type = last_emp.Gender.iloc[0]
                                if Gender_Type == 'Female':
                                    if AssignEscortStatus:
                                        index_values_filtered = filter_data_limit_seats.index.values
                                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                        filter_data_limit_seats['AssignEscort'] = 'Yes'
                                        trip_assign_val = trip_assign_val + 1
                                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    else:
                                        index_values_filtered = filter_data_limit_seats.index.values
                                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                        filter_data_limit_seats['AssignEscort'] = 'No'
                                        trip_assign_val = trip_assign_val + 1
                                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    #drop the data from main
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                index_values_filtered = filter_data_limit_seats.index.values
                                LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                            #reset to actual distance
                            first_val_status = True
                            for index,row in LS_TN_Radius_Original.iterrows():
                                if first_val_status:
                                    ssslon = float(ssslon)
                                    ssslat = float(ssslat)
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    dist_val = 0
                                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                    LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                    first_val_status = False
                                else:
                                    newlat = row.latitude
                                    newlon = row.longitude
                                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                    LS_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                    oldlat = newlat
                                    oldlon = newlon
                                
                            LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                            LS_TN_Radius_Original['DistanceBTWEMP'] = LS_TN_Radius_Original['DistanceBTWEMP'].round(3)
                            
                            timebtw = 0
                            for index,row in LS_TN_Radius_Original.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                LS_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                            
                            timetooffice = 0
                            for index,row in LS_TN_Radius_Original.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                LS_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                            
                    
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val  
            
            
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                        #get the Last employee Details
                        last_emp = filter_data_limit_seats.tail(1)
                        Gender_Type = last_emp.Gender.iloc[0]
                        Last_emp_index = last_emp.index.values
                        if Gender_Type == 'Female':
                            filter_data_limit_seats.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                #drop the data from main
                                index_values_filtered = filter_data_limit_seats.index.values
                                GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        else:
                            index_values_filtered = filter_data_limit_seats.index.values
                            GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            trip_assign_val = trip_assign_val + 1
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                    
                    
                        #reset to actual distance
                        first_val_status = True
                        for index,row in GR_TN_Radius_Original.iterrows():
                            if first_val_status:
                                ssslon = float(ssslon)
                                ssslat = float(ssslat)
                                oldlat = row.latitude
                                oldlon = row.longitude
                                dist_val = 0
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                first_val_status = False
                            else:
                                newlat = row.latitude
                                newlon = row.longitude
                                dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                oldlat = newlat
                                oldlon = newlon
                            
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                        GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].round(3)
                        
                        timebtw = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                        
                        timetooffice = 0
                        for index,row in GR_TN_Radius_Original.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            GR_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
                    
                    else:
                        
                        if len(GR_TN_Radius_Original) == 1:
                            filter_data_limit_seats = GR_TN_Radius_Original.head(1)
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                index_values_filtered = filter_data_limit_seats.index.values
                                GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                    
                        else:
                            filter_data_limit_seats = GR_TN_Radius_Original.head(len(GR_TN_Radius_Original))
                            
                            last_emp = filter_data_limit_seats.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                filter_data_limit_seats.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                last_emp = filter_data_limit_seats.tail(1)
                                Gender_Type = last_emp.Gender.iloc[0]
                                if Gender_Type == 'Female':
                                    if AssignEscortStatus:
                                        index_values_filtered = filter_data_limit_seats.index.values
                                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                        filter_data_limit_seats['AssignEscort'] = 'Yes'
                                        trip_assign_val = trip_assign_val + 1
                                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    else:
                                        index_values_filtered = filter_data_limit_seats.index.values
                                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                        filter_data_limit_seats['AssignEscort'] = 'No'
                                        trip_assign_val = trip_assign_val + 1
                                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                else:
                                    #drop the data from main
                                    index_values_filtered = filter_data_limit_seats.index.values
                                    GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                    GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                    filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                    filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            else:
                                index_values_filtered = filter_data_limit_seats.index.values
                                GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                                GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                            #reset to actual distance
                            first_val_status = True
                            for index,row in GR_TN_Radius_Original.iterrows():
                                if first_val_status:
                                    ssslon = float(ssslon)
                                    ssslat = float(ssslat)
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    dist_val = 0
                                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                                    GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                    first_val_status = False
                                else:
                                    newlat = row.latitude
                                    newlon = row.longitude
                                    dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                                    GR_TN_Radius_Original.at[index,'DistanceBTWEMP'] = dist_val
                                    oldlat = newlat
                                    oldlon = newlon
                                
                            GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].astype(float)
                            GR_TN_Radius_Original['DistanceBTWEMP'] = GR_TN_Radius_Original['DistanceBTWEMP'].round(3)
                            
                            timebtw = 0
                            for index,row in GR_TN_Radius_Original.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                GR_TN_Radius_Original.at[index,'TimeBtwEmp'] = timebtw
                            
                            timetooffice = 0
                            for index,row in GR_TN_Radius_Original.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                GR_TN_Radius_Original.at[index,'TimetoOffice'] = timetooffice 
            

            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_no_validation_NoRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_no_validation_NoRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_no_validation_NoRule", "code": 501}, 501

    @classmethod
    def drop_minutes_rule(cls,filter_data_limit_seats,hour_max_set,ssslat,ssslon): 
        try:
            #do for minutes and No Rule
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    #total minutes taken
                    #Regenerate First Position Proper Distance
                    first_val_status = True
                    for index,row in single_dataset.iterrows():
                        if first_val_status:
                            ssslon = float(ssslon)
                            ssslat = float(ssslat)
                            oldlat = row.latitude
                            oldlon = row.longitude
                            dist_val = 0
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            first_val_status = False
                        else:
                            newlat = row.latitude
                            newlon = row.longitude
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            oldlat = newlat
                            oldlon = newlon
                        
        
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].astype(float)
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].round(3)
                    
                    timebtw = 0
                    for index,row in single_dataset.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        single_dataset.at[index,'TimeBtwEmp'] = timebtw
                    
                    timetooffice = 0
                    for index,row in single_dataset.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        single_dataset.at[index,'TimetoOffice'] = timetooffice 
                
                    TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['TimeBtwEmp']
                    if TotalKMCovered < hour_max_set:
                        #some logic last person
                        if len(single_dataset) == 1:
                            single_object_df['Total_Person'] = len(single_dataset)
                            single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                            single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                            json_data = single_dataset.to_json(orient='records')
                            data_list  = json.loads(json_data)
                            single_object_df['RouteEmployees'] = data_list
                            single_list_emp.append(single_object_df)
                            all_set_emp.append(single_list_emp)      
                        else:
                            filter_last_one = single_dataset.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                single_dataset.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['TimeBtwEmp']
                                single_object_df['Total_Person'] = len(single_dataset)
                                single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                                json_data = single_dataset.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)
                            else:
                                single_object_df['Total_Person'] = len(single_dataset)
                                single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                                json_data = single_dataset.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)      
                    else:
                        #check remaining is one or more then apply logic
                        Total_minutes_covered = 0
                        list_index_val = []
                        for index,row in single_dataset.iterrows(): 
                            Total_minutes_covered = Total_minutes_covered+row.TimeBtwEmp
                            if Total_minutes_covered <= hour_max_set:
                                list_index_val.append(index)
                    
                        if len(list_index_val):
                            single_set_filtered  = single_dataset[single_dataset.index.isin(list_index_val)]
                            if len(single_set_filtered) == 1:
                                TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                                single_object_df['Total_Person'] = len(single_set_filtered)
                                single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                json_data = single_set_filtered.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)
                            else:
                                filter_last_one = single_set_filtered.tail(1)
                                last_emp = filter_last_one.tail(1)
                                Gender_Type = last_emp.Gender.iloc[0]
                                Last_emp_index = last_emp.index.values
                                if Gender_Type == 'Female':
                                    single_set_filtered.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                    TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                                    single_object_df['Total_Person'] = len(single_set_filtered)
                                    single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                                    single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                    json_data = single_set_filtered.to_json(orient='records')
                                    data_list  = json.loads(json_data)
                                    single_object_df['RouteEmployees'] = data_list
                                    single_list_emp.append(single_object_df)
                                    all_set_emp.append(single_list_emp)
                                else:
                                    TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                                    single_object_df['Total_Person'] = len(single_set_filtered)
                                    single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                                    single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                    json_data = single_set_filtered.to_json(orient='records')
                                    data_list  = json.loads(json_data)
                                    single_object_df['RouteEmployees'] = data_list
                                    single_list_emp.append(single_object_df)
                                    all_set_emp.append(single_list_emp)
                        else:
                            single_set_filtered = single_dataset.head(1)
                            TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                            single_object_df['Total_Person'] = len(single_set_filtered)
                            single_object_df['Total_Minutes_Covered'] = TotalKMCovered
                            single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                            json_data = single_set_filtered.to_json(orient='records')
                            data_list  = json.loads(json_data)
                            single_object_df['RouteEmployees'] = data_list
                            single_list_emp.append(single_object_df)
                            all_set_emp.append(single_list_emp)
                    
        
            Total_Emp_Sets_All = []
            for i,x in enumerate(all_set_emp):
                singleset_all = []
                singleset_all.append(i)
                singleset_all.append(x[0]['Total_Person'])
                singleset_all.append(x[0]['Total_Minutes_Covered'])
                Total_Emp_Sets_All.append(singleset_all)
            
            #find the maximum number count from the dict
            max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
            #filter only max_employee count  
            max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
            #filter only minimum data
            max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
            #filter only single list with covers more employee and less km 
            max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
            #filter only perticular set and return it.
            dataaset = all_set_emp[max_people_list[0][0]]
            return dataaset
        except Exception as e:
            logging.error('Failed to execute function drop_minutes_norule: '+ str(e) )
            test_logger.error('Failed to execute function drop_minutes_norule : '+ str(e))
            return {"message": "internal server error, drop_minutes_norule", "code": 501}, 501

    @classmethod
    def Generate_Trip_Hour_WithRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,hour_max_set,ssslat,ssslon,AssignEscortStatus): 
        try:
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_rule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_rule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                else:
                    break
        
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val  
                
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_rule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_minutes_rule(filter_data_limit_seats,hour_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                else:
                    break
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_Hour_WithRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_Hour_WithRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_Hour_WithRule", "code": 501}, 501

    @classmethod
    def drop_KM_rule(cls,filter_data_limit_seats,distance_max_set,ssslat,ssslon): 
        try:
            #do for minutes and No Rule
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    #total minutes taken
                    #Regenerate First Position Proper Distance
                    first_val_status = True
                    for index,row in single_dataset.iterrows():
                        if first_val_status:
                            ssslon = float(ssslon)
                            ssslat = float(ssslat)
                            oldlat = row.latitude
                            oldlon = row.longitude
                            dist_val = 0
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, ssslon, ssslat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            first_val_status = False
                        else:
                            newlat = row.latitude
                            newlon = row.longitude
                            dist_val = CalcDist.distance_two_points(oldlon, oldlat, newlon, newlat)
                            single_dataset.at[index,'DistanceBTWEMP'] = dist_val
                            oldlat = newlat
                            oldlon = newlon
                        
        
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].astype(float)
                    single_dataset['DistanceBTWEMP'] = single_dataset['DistanceBTWEMP'].round(3)
                    
                    timebtw = 0
                    for index,row in single_dataset.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        single_dataset.at[index,'TimeBtwEmp'] = timebtw
                    
                    timetooffice = 0
                    for index,row in single_dataset.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        single_dataset.at[index,'TimetoOffice'] = timetooffice 
                
                    TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                    if TotalKMCovered < distance_max_set:
                        #some logic last person
                        if len(single_dataset) == 1:
                            single_object_df['Total_Person'] = len(single_dataset)
                            single_object_df['Total_KM_Covered'] = TotalKMCovered
                            single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                            json_data = single_dataset.to_json(orient='records')
                            data_list  = json.loads(json_data)
                            single_object_df['RouteEmployees'] = data_list
                            single_list_emp.append(single_object_df)
                            all_set_emp.append(single_list_emp)      
                        else:
                            filter_last_one = single_dataset.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                single_dataset.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                TotalKMCovered = single_dataset.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                                single_object_df['Total_Person'] = len(single_dataset)
                                single_object_df['Total_KM_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                                json_data = single_dataset.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)
                            else:
                                single_object_df['Total_Person'] = len(single_dataset)
                                single_object_df['Total_KM_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_dataset.index.values)
                                json_data = single_dataset.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)      
                    else:
                        #check remaining is one or more then apply logic
                        Total_km_covered = 0
                        list_index_val = []
                        for index,row in single_dataset.iterrows(): 
                            Total_km_covered = Total_km_covered+row.DistanceBTWEMP
                            if Total_km_covered <= distance_max_set:
                                list_index_val.append(index)
                    
                        if len(list_index_val):
                            single_set_filtered  = single_dataset[single_dataset.index.isin(list_index_val)]
                            if len(single_set_filtered) == 1:
                                TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                                single_object_df['Total_Person'] = len(single_set_filtered)
                                single_object_df['Total_KM_Covered'] = TotalKMCovered
                                single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                json_data = single_set_filtered.to_json(orient='records')
                                data_list  = json.loads(json_data)
                                single_object_df['RouteEmployees'] = data_list
                                single_list_emp.append(single_object_df)
                                all_set_emp.append(single_list_emp)
                            else:
                                filter_last_one = single_set_filtered.tail(1)
                                last_emp = filter_last_one.tail(1)
                                Gender_Type = last_emp.Gender.iloc[0]
                                Last_emp_index = last_emp.index.values
                                if Gender_Type == 'Female':
                                    single_set_filtered.drop(labels=None, axis=0, index=Last_emp_index, columns=None, level=None, inplace=True, errors='raise')
                                    TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                                    single_object_df['Total_Person'] = len(single_set_filtered)
                                    single_object_df['Total_KM_Covered'] = TotalKMCovered
                                    single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                    json_data = single_set_filtered.to_json(orient='records')
                                    data_list  = json.loads(json_data)
                                    single_object_df['RouteEmployees'] = data_list
                                    single_list_emp.append(single_object_df)
                                    all_set_emp.append(single_list_emp)
                                else:
                                    TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                                    single_object_df['Total_Person'] = len(single_set_filtered)
                                    single_object_df['Total_KM_Covered'] = TotalKMCovered
                                    single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                                    json_data = single_set_filtered.to_json(orient='records')
                                    data_list  = json.loads(json_data)
                                    single_object_df['RouteEmployees'] = data_list
                                    single_list_emp.append(single_object_df)
                                    all_set_emp.append(single_list_emp)
                        else:
                            single_set_filtered = single_dataset.head(1)
                            TotalKMCovered = single_set_filtered.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                            single_object_df['Total_Person'] = len(single_set_filtered)
                            single_object_df['Total_KM_Covered'] = TotalKMCovered
                            single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                            json_data = single_set_filtered.to_json(orient='records')
                            data_list  = json.loads(json_data)
                            single_object_df['RouteEmployees'] = data_list
                            single_list_emp.append(single_object_df)
                            all_set_emp.append(single_list_emp)
                    
        
            Total_Emp_Sets_All = []
            for i,x in enumerate(all_set_emp):
                singleset_all = []
                singleset_all.append(i)
                singleset_all.append(x[0]['Total_Person'])
                singleset_all.append(x[0]['Total_KM_Covered'])
                Total_Emp_Sets_All.append(singleset_all)
            
            #find the maximum number count from the dict
            max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
            #filter only max_employee count  
            max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
            #filter only minimum data
            max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
            #filter only single list with covers more employee and less km 
            max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
            #filter only perticular set and return it.
            dataaset = all_set_emp[max_people_list[0][0]]
            return dataaset
        except Exception as e:
            logging.error('Failed to execute function drop_KM_rule: '+ str(e) )
            test_logger.error('Failed to execute function drop_KM_rule : '+ str(e))
            return {"message": "internal server error, drop_KM_rule", "code": 501}, 501

    @classmethod
    def Generate_Trip_KM_WithRule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,distance_max_set,ssslat,ssslon,AssignEscortStatus): 
        try:
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_KM_rule(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_KM_rule(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                LS_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(LS_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                else:
                    break
        
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val  
                
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_KM_rule(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.drop_KM_rule(filter_data_limit_seats,distance_max_set,ssslat,ssslon)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        
                        if len(data_for_trip_assign) == 1:
                            filter_last_one = data_for_trip_assign.head(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                        else:
                            filter_last_one = data_for_trip_assign.tail(1)
                            last_emp = filter_last_one.tail(1)
                            Gender_Type = last_emp.Gender.iloc[0]
                            Last_emp_index = last_emp.index.values
                            if Gender_Type == 'Female':
                                if AssignEscortStatus:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'Yes'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                                else:
                                    data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                    data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                    data_for_trip_assign['AssignEscort'] = 'No'
                                    trip_assign_val = trip_assign_val + 1
                                    Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                    #drop the vehicle index from original sets
                                    vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                    #drop the assigned employee from main set
                                    GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                            else:
                                data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                                data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                                data_for_trip_assign['AssignEscort'] = 'No'
                                trip_assign_val = trip_assign_val + 1
                                Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                                #drop the vehicle index from original sets
                                vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                                #drop the assigned employee from main set
                                GR_TN_Radius_Original = CalcDist.Drop_Index_and_Reset(GR_TN_Radius_Original,index_values_assigning,ssslat,ssslon)        
                else:
                    break
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function Generate_Trip_KM_WithRule: '+ str(e) )
            test_logger.error('Failed to execute function Generate_Trip_KM_WithRule : '+ str(e))
            return {"message": "internal server error, Generate_Trip_KM_WithRule", "code": 501}, 501





    @classmethod
    def add_to_dataframe_only_two_records_lesser(cls,filtered_data,facility_stop_data,linestring_df2,occur_val,MaxEmplPickupRadius):
        try:
            km_val = 0
            max_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()]
            min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()]
            
            if len(max_data) == 2 and len(min_data) == 2:
                uniquelatlon = filtered_data.LatLng.unique()
                uniquelatlonlist = uniquelatlon.tolist()
                if len(uniquelatlonlist) == 1:
                    #add first Data
                    min_data = filtered_data[:1]
                    listToStr = ' '.join(map(str, uniquelatlonlist)) 
                    my_list = listToStr.split(",")
                    min_data['closest_stop_geom'] = my_list[0]+","+my_list[1]
                    min_data['Route'] = occur_val
                    min_data['KM_distance'] = 0
                    min_data['fac_latitude'] = my_list[0]
                    min_data['fac_longitude'] = my_list[1]
                    min_data['distance'] = 0
                    linestring_df2 = linestring_df2.append(min_data, ignore_index=True)
                    #add second data
                    max_data = filtered_data[1:]
                    max_data['Route'] = occur_val
                    linestring_df2 = linestring_df2.append(max_data, ignore_index=True)
                    occur_val = occur_val + 1
                else:
                    max_data = filtered_data[1:]
                    mlat = max_data.latitude.iloc[0]
                    mlon = max_data.longitude.iloc[0]
                    newer_data = mlat.astype(str)+","+mlon.astype(str)
                    stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                    
                    min_data = filtered_data[:1]
                    min_data_one = min_data.copy()
                    min_data.drop(['fac_latitude','fac_longitude', 'distance','closest_stop_geom','KM_distance'], axis=1, inplace=True)
                    min_filtered_data = CalcDist.distance_data(stop_data_near,min_data)
                    
                    km_value = min_filtered_data.KM_distance.iloc[0]
                    km_val = km_val + km_value
                    if km_val < MaxEmplPickupRadius:
                        min_filtered_data['Route'] = occur_val
                        linestring_df2 = linestring_df2.append(min_filtered_data, ignore_index=True)
                        max_data['Route'] = occur_val
                        linestring_df2 = linestring_df2.append(max_data, ignore_index=True)
                        occur_val = occur_val + 1
                    else:
                        min_data_one['Route'] = occur_val
                        linestring_df2 = linestring_df2.append(min_data_one, ignore_index=True)
                        occur_val = occur_val + 1
                        max_data['Route'] = occur_val
                        linestring_df2 = linestring_df2.append(max_data, ignore_index=True)
                        occur_val = occur_val + 1    
            else:
               
                min_data_one = min_data.copy()
                mlat = max_data.latitude.iloc[0]
                mlon = max_data.longitude.iloc[0]
                newer_data = mlat.astype(str)+","+mlon.astype(str)
        
                #add the minimum data into the route
                stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                min_data.drop(['fac_latitude','fac_longitude', 'distance', 'closest_stop_geom', 'KM_distance'], axis=1, inplace=True)
                min_filtered_data = CalcDist.distance_data(stop_data_near,min_data)
                km_value = min_filtered_data.KM_distance.iloc[0]
                km_val = km_val + km_value

                
        
                if km_val < MaxEmplPickupRadius:
                    min_filtered_data['Route'] = occur_val
                    linestring_df2 = linestring_df2.append(min_filtered_data, ignore_index=True)
                    max_data['Route'] = occur_val
                    linestring_df2 = linestring_df2.append(max_data, ignore_index=True)
                    occur_val = occur_val + 1
                else:
                    min_data_one['Route'] = occur_val
                    linestring_df2 = linestring_df2.append(min_data_one, ignore_index=True)
                    occur_val = occur_val + 1
                    max_data['Route'] = occur_val
                    linestring_df2 = linestring_df2.append(max_data, ignore_index=True)
                    occur_val = occur_val + 1    
            return linestring_df2
        except Exception as e:
            logging.error('Failed to execute function add_to_dataframe_only_two_records_lesser: '+ str(e) )
            test_logger.error('Failed to execute function add_to_dataframe_only_two_records_lesser : '+ str(e))
            return {"message": "internal server error, add_to_dataframe_only_two_records_lesser", "code": 501}, 501  
            


    @classmethod
    def generate_pick_no_rule(cls,GR_TN_Radius_Original,LS_TN_Radius_Original,vehicle_data_original_set,stop_lat,stop_lon,facility_stop_data): 
        try:
            columns_pick = ['ID', 'Name', 'LatLng', 
                                    'Gender', 'latitude',
            'longitude','DistanceToOffice','PeakSpeed',
            'Vehicle_ID','Vehicle_Trips','AssignEscort']
            
            assign_colsss = ['ID', 'Name', 'LatLng', 'Gender', 'latitude', 'longitude',
            'DistanceToOffice', 'PeakSpeed', 'Vehicle_ID', 'Vehicle_Trips',
            'AssignEscort', 'fac_latitude', 'fac_longitude', 'distance',
            'closest_stop_geom', 'KM_distance']
            
            Trip_Data_for_Drop = pd.DataFrame(columns = columns_pick) 
            
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick) 
                seat_count = row.Seats
                vehicle_ID = row.ID
                
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        filter_data_limit_seats['AssignEscort'] = 'No'
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        #totaltimetaken
                        # total_time = filter_data_limit_seats.sum(axis = 0, skipna = True)['TimeBtwEmp']
                        # timetooffice = filter_data_limit_seats.tail(1)['TimetoOffice'].iloc[0]
                        # total_trip_time = total_time + timetooffice + int(Buffer_time)
                        
                        #qdd to the dataframe
                        index_values_filtered = filter_data_limit_seats.index.values
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        
                        #drop the unneccessary columns
                        
                        #assign pickorder
                        filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                        ]]
                        #generate Route and Pick Order
                        loop_count = 1
                        for index,row in filter_data_limit_seats.iterrows():
                            if len(filter_data_limit_seats) > 0:
                                if loop_count == 1:
                                    filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                    one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                    maxlat = one_data.latitude.iloc[0]
                                    maxlon = one_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    
                                    index_pos = one_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                
                                    if len(filter_data_limit_seats) == 1:
                                        mlat = one_data.latitude.iloc[0]
                                        mlon = one_data.longitude.iloc[0]
                                        newer_data = mlat.astype(str)+","+mlon.astype(str)
                                        stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                        min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                        index_pos = min_filtered_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    else:
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        loop_count = loop_count + 1
                                else:
                                    filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                    min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                    maxlat = min_data.latitude.iloc[0]
                                    maxlon = min_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                    index_pos = min_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    loop_count = loop_count + 1
                            else:
                                loop_count = 1
                        
                        
                        #assign in order closest latlon
                        Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                        first_statusss_val = True
                        for index,row in Trip_Data_for_Drop_ordered.iterrows():
                            if first_statusss_val:
                                index__vall = index
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                first_statusss_val = False
                            else:
                                oldlat = row.latitude
                                oldlon = row.longitude
                                mergedlatlon = str(oldlat)+','+str(oldlon)
                                Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                index__vall = index
                        
                        #get the last data    
                        last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                        getlastindex = last_data.index.values
                        fac_latlon = str(stop_lat)+','+str(stop_lon)
                        Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                        
                        #drop and resgenerate KM_distance
                        Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                        
                        #calculate the distance
                        lat = []
                        lon = []
                        for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                            # Try to,
                            try:
                                lat.append(row.split(',')[0])
                                lon.append(row.split(',')[1])
                            except:
                                lat.append(np.NaN)
                                lon.append(np.NaN)  
                    
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                        
                        
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                        #generate distance btwn and timebtn and timetooffice
                        Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                        #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                        
                        
                        #create a new derived column called TimeBtwEmp
                        timebtw = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                            
                        #create a new derived column called TimetoOffice
                        timetooffice = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                        
                        
                        #append it
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                        
                        
                    else:
                        
                        
                        if len(LS_TN_Radius_Original) == 1:
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            filter_data_limit_seats = LS_TN_Radius_Original.copy()
                            index_values_filtered = filter_data_limit_seats.index.values
                            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            filter_data_limit_seats['closest_stop_geom'] = fac_latlon
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort','closest_stop_geom'
                                        ]]
                            lat = []
                            lon = []
                            for row in filter_data_limit_seats['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            filter_data_limit_seats['fac_latitude'] = lat
                            filter_data_limit_seats['fac_longitude'] = lon  
                            filter_data_limit_seats['fac_latitude'] = filter_data_limit_seats['fac_latitude'].astype(float)
                            filter_data_limit_seats['fac_longitude'] = filter_data_limit_seats['fac_longitude'].astype(float)
                            
                            for index,row in filter_data_limit_seats.iterrows():
                                filter_data_limit_seats.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                            #generate distance btwn and timebtn and timetooffice
                            filter_data_limit_seats['DistanceBTWEMP'] = filter_data_limit_seats['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        else:
                            filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            #qdd to the dataframe
                            index_values_filtered = filter_data_limit_seats.index.values
                            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                            
                            #drop the unneccessary columns
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed',
                                            'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                            ]]
                            #generate Route and Pick Order
                            loop_count = 1
                            for index,row in filter_data_limit_seats.iterrows():
                                if len(filter_data_limit_seats) > 0:
                                    if loop_count == 1:
                                        filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                        one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                        maxlat = one_data.latitude.iloc[0]
                                        maxlon = one_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        
                                        index_pos = one_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    
                                        if len(filter_data_limit_seats) == 1:
                                            mlat = one_data.latitude.iloc[0]
                                            mlon = one_data.longitude.iloc[0]
                                            newer_data = mlat.astype(str)+","+mlon.astype(str)
                                            stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                            min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                            index_pos = min_filtered_data.index.values
                                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                            filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        else:
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            loop_count = loop_count + 1
                                    else:
                                        filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                        min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                        maxlat = min_data.latitude.iloc[0]
                                        maxlon = min_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                        index_pos = min_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        loop_count = loop_count + 1
                                else:
                                    loop_count = 1
                            
                            
                            #assign in order closest latlon
                            Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                            first_statusss_val = True
                            for index,row in Trip_Data_for_Drop_ordered.iterrows():
                                if first_statusss_val:
                                    index__vall = index
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    first_statusss_val = False
                                else:
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    mergedlatlon = str(oldlat)+','+str(oldlon)
                                    Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    index__vall = index
                            
                            #get the last data    
                            last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            
                            #drop and resgenerate KM_distance
                            Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                            
                            #calculate the distance
                            lat = []
                            lon = []
                            for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                            
                            
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                            
                            
                            #generate distance btwn and timebtn and timetooffice
                            Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
    
                else:
                    break
            
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val
            
            
            for index,row in vehicle_data_original_set.iterrows(): 
                Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick) 
                seat_count = row.Seats
                vehicle_ID = row.ID
                
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                        filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                        filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        filter_data_limit_seats['AssignEscort'] = 'No'
                        trip_assign_val = trip_assign_val + 1
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        #totaltimetaken
                        # total_time = filter_data_limit_seats.sum(axis = 0, skipna = True)['TimeBtwEmp']
                        # timetooffice = filter_data_limit_seats.tail(1)['TimetoOffice'].iloc[0]
                        # total_trip_time = total_time + timetooffice + int(Buffer_time)
                        
                        #qdd to the dataframe
                        index_values_filtered = filter_data_limit_seats.index.values
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        
                        #drop the unneccessary columns
                        
                        #assign pickorder
                        filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                        ]]
                        #generate Route and Pick Order
                        loop_count = 1
                        for index,row in filter_data_limit_seats.iterrows():
                            if len(filter_data_limit_seats) > 0:
                                if loop_count == 1:
                                    filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                    one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                    maxlat = one_data.latitude.iloc[0]
                                    maxlon = one_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    
                                    index_pos = one_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                
                                    if len(filter_data_limit_seats) == 1:
                                        mlat = one_data.latitude.iloc[0]
                                        mlon = one_data.longitude.iloc[0]
                                        newer_data = mlat.astype(str)+","+mlon.astype(str)
                                        stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                        min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                        index_pos = min_filtered_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    else:
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        loop_count = loop_count + 1
                                else:
                                    filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                    min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                    maxlat = min_data.latitude.iloc[0]
                                    maxlon = min_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                    index_pos = min_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    loop_count = loop_count + 1
                            else:
                                loop_count = 1
                        
                        
                        #assign in order closest latlon
                        Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                        first_statusss_val = True
                        for index,row in Trip_Data_for_Drop_ordered.iterrows():
                            if first_statusss_val:
                                index__vall = index
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                first_statusss_val = False
                            else:
                                oldlat = row.latitude
                                oldlon = row.longitude
                                mergedlatlon = str(oldlat)+','+str(oldlon)
                                Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                index__vall = index
                        
                        #get the last data    
                        last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                        getlastindex = last_data.index.values
                        fac_latlon = str(stop_lat)+','+str(stop_lon)
                        Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                        
                        #drop and resgenerate KM_distance
                        Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                        
                        #calculate the distance
                        lat = []
                        lon = []
                        for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                            # Try to,
                            try:
                                lat.append(row.split(',')[0])
                                lon.append(row.split(',')[1])
                            except:
                                lat.append(np.NaN)
                                lon.append(np.NaN)  
                    
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                        
                        
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                        #generate distance btwn and timebtn and timetooffice
                        Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                        #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                        
                        
                        #create a new derived column called TimeBtwEmp
                        timebtw = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                            
                        #create a new derived column called TimetoOffice
                        timetooffice = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                        
                        
                        #append it
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                        
                        
                    else:
                        
                        
                        if len(GR_TN_Radius_Original) == 1:
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            filter_data_limit_seats = GR_TN_Radius_Original.copy()
                            index_values_filtered = filter_data_limit_seats.index.values
                            GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            filter_data_limit_seats['closest_stop_geom'] = fac_latlon
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort','closest_stop_geom'
                                        ]]
                            lat = []
                            lon = []
                            for row in filter_data_limit_seats['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            filter_data_limit_seats['fac_latitude'] = lat
                            filter_data_limit_seats['fac_longitude'] = lon  
                            filter_data_limit_seats['fac_latitude'] = filter_data_limit_seats['fac_latitude'].astype(float)
                            filter_data_limit_seats['fac_longitude'] = filter_data_limit_seats['fac_longitude'].astype(float)
                            
                            for index,row in filter_data_limit_seats.iterrows():
                                filter_data_limit_seats.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                            #generate distance btwn and timebtn and timetooffice
                            filter_data_limit_seats['DistanceBTWEMP'] = filter_data_limit_seats['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        else:
                            filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            #qdd to the dataframe
                            index_values_filtered = filter_data_limit_seats.index.values
                            GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                            
                            #drop the unneccessary columns
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed',
                                            'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                            ]]
                            #generate Route and Pick Order
                            loop_count = 1
                            for index,row in filter_data_limit_seats.iterrows():
                                if len(filter_data_limit_seats) > 0:
                                    if loop_count == 1:
                                        filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                        one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                        maxlat = one_data.latitude.iloc[0]
                                        maxlon = one_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        
                                        index_pos = one_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    
                                        if len(filter_data_limit_seats) == 1:
                                            mlat = one_data.latitude.iloc[0]
                                            mlon = one_data.longitude.iloc[0]
                                            newer_data = mlat.astype(str)+","+mlon.astype(str)
                                            stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                            min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                            index_pos = min_filtered_data.index.values
                                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                            filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        else:
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            loop_count = loop_count + 1
                                    else:
                                        filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                        min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                        maxlat = min_data.latitude.iloc[0]
                                        maxlon = min_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                        index_pos = min_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        loop_count = loop_count + 1
                                else:
                                    loop_count = 1
                            
                            
                            #assign in order closest latlon
                            Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                            first_statusss_val = True
                            for index,row in Trip_Data_for_Drop_ordered.iterrows():
                                if first_statusss_val:
                                    index__vall = index
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    first_statusss_val = False
                                else:
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    mergedlatlon = str(oldlat)+','+str(oldlon)
                                    Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    index__vall = index
                            
                            #get the last data    
                            last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            
                            #drop and resgenerate KM_distance
                            Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                            
                            #calculate the distance
                            lat = []
                            lon = []
                            for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                            
                            
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                            
                            
                            #generate distance btwn and timebtn and timetooffice
                            Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
    
                else:
                    break
            
            
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_no_rule: '+ str(e) )
            return {"message": "internal server error, generate_pick_no_rule", "code": 501}, 501



    @classmethod
    def generate_pick_hour_no_rule(cls,filter_data_limit_seats,hour_max_set,stop_lat,stop_lon,facility_stop_data,Buffer_time): 
        try:
            #do for minutes and No Rule
            columns_pick_ordered = ['ID', 'Name', 'LatLng', 
                                                        'Gender', 'latitude',
                                'longitude','DistanceToOffice','PeakSpeed']
            
            
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick_ordered) 
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    
                    single_data_index_values = single_dataset.index.values
                    
                    single_dataset = single_dataset[['ID', 'Name', 'LatLng', 
                                            'Gender', 'latitude',
                    'longitude','DistanceToOffice','PeakSpeed'
                    ]]
                    
                    loop_count = 1
                    for index,row in single_dataset.iterrows():
                        if len(single_dataset) > 0:
                            if loop_count == 1:
                                filtered_data = CalcDist.distance_data(facility_stop_data,single_dataset)
                                one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                maxlat = one_data.latitude.iloc[0]
                                maxlon = one_data.longitude.iloc[0]
                                stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                
                                index_pos = one_data.index.values
                                single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                #single_dataset = single_dataset.reset_index(drop = True)
                            
                                if len(single_dataset) == 1:
                                    mlat = one_data.latitude.iloc[0]
                                    mlon = one_data.longitude.iloc[0]
                                    newer_data = mlat.astype(str)+","+mlon.astype(str)
                                    stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                    min_filtered_data = CalcDist.distance_data(stop_data_near,single_dataset)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=False)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=False)
                                    index_pos = min_filtered_data.index.values
                                    single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    #single_dataset = single_dataset.reset_index(drop = True)
                                else:
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=False)
                                    loop_count = loop_count + 1
                            else:
                                filtered_data = CalcDist.distance_data(stopss_copy,single_dataset)
                                min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                maxlat = min_data.latitude.iloc[0]
                                maxlon = min_data.longitude.iloc[0]
                                stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=False)
                                index_pos = min_data.index.values
                                single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                #single_dataset = single_dataset.reset_index(drop = True)
                                loop_count = loop_count + 1
                        else:
                            loop_count = 1
                        
                    #assign in order closest latlon
                    Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=columns_pick_ordered)
                    first_statusss_val = True
                    for index,row in Trip_Data_for_Drop_ordered.iterrows():
                        if first_statusss_val:
                            index__vall = index
                            Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=False)
                            first_statusss_val = False
                        else:
                            oldlat = row.latitude
                            oldlon = row.longitude
                            mergedlatlon = str(oldlat)+','+str(oldlon)
                            Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                            Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=False)
                            index__vall = index
                    
                    #get the last data    
                    last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                    getlastindex = last_data.index.values
                    fac_latlon = str(stop_lat)+','+str(stop_lon)
                    Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon   
                    
                    #drop and resgenerate KM_distance
                    Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                    
                    #calculate the distance
                    lat = []
                    lon = []
                    for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                    # Try to,
                        try:
                            lat.append(row.split(',')[0])
                            lon.append(row.split(',')[1])
                        except:
                            lat.append(np.NaN)
                            lon.append(np.NaN)  
                    
                    Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                    Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                    Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                    Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                    
                    
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                    #generate distance btwn and timebtn and timetooffice
                    Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                    #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                    
                    
                    #create a new derived column called TimeBtwEmp
                    timebtw = 0
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                    
                    #create a new derived column called TimetoOffice
                    timetooffice = 0
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                    
                    totalmncovered = Trip_Data_for_Drop_ordered_new.sum(axis = 0, skipna = True)['TimeBtwEmp']
                    totalmncoveredbuffertime = totalmncovered + int(Buffer_time)
                    
                    if totalmncoveredbuffertime < hour_max_set:
                        #continue
                        single_object_df['Total_Person'] = len(Trip_Data_for_Drop_ordered_new)
                        single_object_df['Total_Minutes_Covered'] = totalmncovered
                        single_object_df['All_Emp_Index'] = list(single_data_index_values)
                        json_data = Trip_Data_for_Drop_ordered_new.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                    else:
                        
                        #contnue
                        Total_minutes_covered = 0
                        list_index_val = []
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows(): 
                            Total_minutes_covered = Total_minutes_covered+row.TimeBtwEmp
                            Total_minutes_covered_Buffer_time = Total_minutes_covered + int(Buffer_time)
                            if Total_minutes_covered_Buffer_time <= hour_max_set:
                                list_index_val.append(index)
                        
                        if len(list_index_val):
                            single_set_filtered  = Trip_Data_for_Drop_ordered_new[Trip_Data_for_Drop_ordered_new.index.isin(list_index_val)]
                        
                            last_data = single_set_filtered.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            single_set_filtered.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            single_set_filtered.at[getlastindex,'fac_latitude'] = float(stop_lat) 
                            single_set_filtered.at[getlastindex,'fac_longitude'] = float(stop_lon)
                            
                            for index,row in single_set_filtered.iterrows():
                                single_set_filtered.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                            #generate distance btwn and timebtn and timetooffice
                            single_set_filtered['DistanceBTWEMP'] = single_set_filtered['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in single_set_filtered.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimeBtwEmp'] = timebtw
                            
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in single_set_filtered.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimetoOffice'] = timetooffice 
                            
                
                        else:
                            single_set_filtered = Trip_Data_for_Drop_ordered_new.head(1)
                        
                            last_data = single_set_filtered.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            single_set_filtered.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            single_set_filtered.at[getlastindex,'fac_latitude'] = float(stop_lat) 
                            single_set_filtered.at[getlastindex,'fac_longitude'] = float(stop_lon)
                            
                            for index,row in single_set_filtered.iterrows():
                                single_set_filtered.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                            #generate distance btwn and timebtn and timetooffice
                            single_set_filtered['DistanceBTWEMP'] = single_set_filtered['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in single_set_filtered.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimeBtwEmp'] = timebtw
                            
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in single_set_filtered.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimetoOffice'] = timetooffice 
            
                        totalmncovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                        single_object_df['Total_Person'] = len(single_set_filtered)
                        single_object_df['Total_Minutes_Covered'] = totalmncovered
                        single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                        json_data = single_set_filtered.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                        
                Total_Emp_Sets_All = []
                for i,x in enumerate(all_set_emp):
                    singleset_all = []
                    singleset_all.append(i)
                    singleset_all.append(x[0]['Total_Person'])
                    singleset_all.append(x[0]['Total_Minutes_Covered'])
                    Total_Emp_Sets_All.append(singleset_all)
                
                #find the maximum number count from the dict
                max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
                #filter only max_employee count  
                max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
                #filter only minimum data
                max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
                #filter only single list with covers more employee and less km 
                max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
                #filter only perticular set and return it.
                dataaset = all_set_emp[max_people_list[0][0]]
            return dataaset    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_hour_no_rule: '+ str(e) )
            test_logger.error('Failed to execute function generate_pick_hour_no_rule : '+ str(e))
            return {"message": "internal server error, generate_pick_hour_no_rule", "code": 501}, 501


    @classmethod
    def generate_pick_no_rule_hour(cls,GR_TN_Radius_Original,LS_TN_Radius_Original,vehicle_data_original_set,stop_lat,stop_lon,hour_max_set,facility_stop_data,Buffer_time): 
        try:
            columns_pick = ['ID', 'Name', 'LatLng', 
                                'Gender', 'latitude',
            'longitude','DistanceToOffice','PeakSpeed',
            'Vehicle_ID','Vehicle_Trips','AssignEscort']
            
            assign_colsss = ['ID', 'Name', 'LatLng', 'Gender', 'latitude', 'longitude',
            'DistanceToOffice', 'PeakSpeed', 'Vehicle_ID', 'Vehicle_Trips',
            'AssignEscort', 'fac_latitude', 'fac_longitude', 'distance',
            'closest_stop_geom', 'KM_distance']
            
            Trip_Data_for_Drop = pd.DataFrame(columns = columns_pick)  
            
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_hour_no_rule(filter_data_limit_seats,hour_max_set,stop_lat,stop_lon,facility_stop_data,Buffer_time)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        print(LS_TN_Radius_Original.shape)
                    else:
                        
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_hour_no_rule(filter_data_limit_seats,hour_max_set,stop_lat,stop_lon,facility_stop_data,Buffer_time)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        
                else:
                    break
            
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val
                
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_hour_no_rule(filter_data_limit_seats,hour_max_set,stop_lat,stop_lon,facility_stop_data,Buffer_time)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        print(GR_TN_Radius_Original.shape)
                    else:
                        
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_hour_no_rule(filter_data_limit_seats,hour_max_set,stop_lat,stop_lon,facility_stop_data,Buffer_time)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        
                else:
                    break
            
            
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_no_rule_hour: '+ str(e) )
            test_logger.error('Failed to execute function generate_pick_no_rule_hour : '+ str(e))
            return {"message": "internal server error, generate_pick_no_rule_hour", "code": 501}, 501



    @classmethod
    def generate_pick_km_no_rule(cls,filter_data_limit_seats,distance_max_set,stop_lat,stop_lon,facility_stop_data): 
        try:
            #do for minutes and No Rule
            columns_pick_ordered = ['ID', 'Name', 'LatLng', 
                                                        'Gender', 'latitude',
                                'longitude','DistanceToOffice','PeakSpeed']
            
            all_set_emp = []
            for i in filter_data_limit_seats:
                if len(i):
                    Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick_ordered) 
                    single_dataset = i.copy()
                    single_list_emp = []
                    single_object_df = {}
                    
                    single_data_index_values = single_dataset.index.values

                    single_dataset = single_dataset[['ID', 'Name', 'LatLng', 
                                            'Gender', 'latitude',
                    'longitude','DistanceToOffice','PeakSpeed'
                    ]]
                    
                    
                    loop_count = 1
                    for index,row in single_dataset.iterrows():
                        if len(single_dataset) > 0:
                            if loop_count == 1:
                                filtered_data = CalcDist.distance_data(facility_stop_data,single_dataset)
                                one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                maxlat = one_data.latitude.iloc[0]
                                maxlon = one_data.longitude.iloc[0]
                                stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                
                                index_pos = one_data.index.values
                                single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                #single_dataset = single_dataset.reset_index(drop = True)
                            
                                if len(single_dataset) == 1:
                                    mlat = one_data.latitude.iloc[0]
                                    mlon = one_data.longitude.iloc[0]
                                    newer_data = mlat.astype(str)+","+mlon.astype(str)
                                    stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                    min_filtered_data = CalcDist.distance_data(stop_data_near,single_dataset)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=False)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=False)
                                    index_pos = min_filtered_data.index.values
                                    single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    #single_dataset = single_dataset.reset_index(drop = True)
                                else:
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=False)
                                    loop_count = loop_count + 1
                            else:
                                filtered_data = CalcDist.distance_data(stopss_copy,single_dataset)
                                min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                maxlat = min_data.latitude.iloc[0]
                                maxlon = min_data.longitude.iloc[0]
                                stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=False)
                                index_pos = min_data.index.values
                                single_dataset.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                #single_dataset = single_dataset.reset_index(drop = True)
                                loop_count = loop_count + 1
                        else:
                            loop_count = 1
                    
                    Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=columns_pick_ordered)
                    first_statusss_val = True
                    for index,row in Trip_Data_for_Drop_ordered.iterrows():
                        if first_statusss_val:
                            index__vall = index
                            Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=False)
                            first_statusss_val = False
                        else:
                            oldlat = row.latitude
                            oldlon = row.longitude
                            mergedlatlon = str(oldlat)+','+str(oldlon)
                            Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                            Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=False)
                            index__vall = index
                    
                    #get the last data    
                    last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                    getlastindex = last_data.index.values
                    fac_latlon = str(stop_lat)+','+str(stop_lon)
                    Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon   
                    
                    #drop and resgenerate KM_distance
                    Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                    
                    #calculate the distance
                    lat = []
                    lon = []
                    for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                    # Try to,
                        try:
                            lat.append(row.split(',')[0])
                            lon.append(row.split(',')[1])
                        except:
                            lat.append(np.NaN)
                            lon.append(np.NaN)  
                    
                    Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                    Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                    Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                    Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                    
                    
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                    #generate distance btwn and timebtn and timetooffice
                    Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                    #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                    
                    
                    #create a new derived column called TimeBtwEmp
                    timebtw = 0
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                        Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                    
                    #create a new derived column called TimetoOffice
                    timetooffice = 0
                    for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                        timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                        Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 

                    totalkmcovered = Trip_Data_for_Drop_ordered_new.sum(axis = 0, skipna = True)['DistanceBTWEMP']
                    
                    if totalkmcovered < distance_max_set:
                        #continue
                        single_object_df['Total_Person'] = len(Trip_Data_for_Drop_ordered_new)
                        single_object_df['Total_KM_Covered'] = totalkmcovered
                        single_object_df['All_Emp_Index'] = list(single_data_index_values)
                        json_data = Trip_Data_for_Drop_ordered_new.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                    else:
                        
                        #contnue
                        Total_km_covered = 0
                        list_index_val = []
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows(): 
                            Total_km_covered = Total_km_covered+row.DistanceBTWEMP
                            if Total_km_covered <= distance_max_set:
                                list_index_val.append(index)
                        
                        if len(list_index_val):
                            single_set_filtered  = Trip_Data_for_Drop_ordered_new[Trip_Data_for_Drop_ordered_new.index.isin(list_index_val)]
                        
                            last_data = single_set_filtered.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            single_set_filtered.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            single_set_filtered.at[getlastindex,'fac_latitude'] = float(stop_lat) 
                            single_set_filtered.at[getlastindex,'fac_longitude'] = float(stop_lon)
                            
                            for index,row in single_set_filtered.iterrows():
                                single_set_filtered.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                            #generate distance btwn and timebtn and timetooffice
                            single_set_filtered['DistanceBTWEMP'] = single_set_filtered['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in single_set_filtered.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimeBtwEmp'] = timebtw
                            
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in single_set_filtered.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimetoOffice'] = timetooffice 
                            
                
                        else:
                            single_set_filtered = Trip_Data_for_Drop_ordered_new.head(1)
                        
                            last_data = single_set_filtered.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            single_set_filtered.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            single_set_filtered.at[getlastindex,'fac_latitude'] = float(stop_lat) 
                            single_set_filtered.at[getlastindex,'fac_longitude'] = float(stop_lon)
                            
                            for index,row in single_set_filtered.iterrows():
                                single_set_filtered.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                    
                    
                            #generate distance btwn and timebtn and timetooffice
                            single_set_filtered['DistanceBTWEMP'] = single_set_filtered['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in single_set_filtered.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimeBtwEmp'] = timebtw
                            
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in single_set_filtered.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                single_set_filtered.at[index,'TimetoOffice'] = timetooffice 
            
                        totalkmcovered = single_set_filtered.sum(axis = 0, skipna = True)['TimeBtwEmp']
                        single_object_df['Total_Person'] = len(single_set_filtered)
                        single_object_df['Total_KM_Covered'] = totalkmcovered
                        single_object_df['All_Emp_Index'] = list(single_set_filtered.index.values)
                        json_data = single_set_filtered.to_json(orient='records')
                        data_list  = json.loads(json_data)
                        single_object_df['RouteEmployees'] = data_list
                        single_list_emp.append(single_object_df)
                        all_set_emp.append(single_list_emp)
                
                Total_Emp_Sets_All = []
                for i,x in enumerate(all_set_emp):
                    singleset_all = []
                    singleset_all.append(i)
                    singleset_all.append(x[0]['Total_Person'])
                    singleset_all.append(x[0]['Total_KM_Covered'])
                    Total_Emp_Sets_All.append(singleset_all)
                
                #find the maximum number count from the dict
                max_people = max([sublist[-2] for sublist in Total_Emp_Sets_All])    
                #filter only max_employee count  
                max_people_list = [c_list for c_list in Total_Emp_Sets_All if max_people == c_list[1]]
                #filter only minimum data
                max_people_with_min_km = min([sublist[-1] for sublist in max_people_list])
                #filter only single list with covers more employee and less km 
                max_people_list = [n_list for n_list in max_people_list if max_people_with_min_km in n_list]
                #filter only perticular set and return it.
                dataaset = all_set_emp[max_people_list[0][0]]
            
            return dataaset    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_km_no_rule: '+ str(e) )
            test_logger.error('Failed to execute function generate_pick_km_no_rule : '+ str(e))
            return {"message": "internal server error, generate_pick_km_no_rule", "code": 501}, 501


    @classmethod
    def generate_pick_no_rule_km(cls,GR_TN_Radius_Original,LS_TN_Radius_Original,vehicle_data_original_set,stop_lat,stop_lon,distance_max_set,facility_stop_data,Buffer_time): 
        try:
            columns_pick = ['ID', 'Name', 'LatLng', 
                                'Gender', 'latitude',
            'longitude','DistanceToOffice','PeakSpeed',
            'Vehicle_ID','Vehicle_Trips','AssignEscort']
            
            assign_colsss = ['ID', 'Name', 'LatLng', 'Gender', 'latitude', 'longitude',
            'DistanceToOffice', 'PeakSpeed', 'Vehicle_ID', 'Vehicle_Trips',
            'AssignEscort', 'fac_latitude', 'fac_longitude', 'distance',
            'closest_stop_geom', 'KM_distance']
            
            Trip_Data_for_Drop = pd.DataFrame(columns = columns_pick)  
            
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_km_no_rule(filter_data_limit_seats,distance_max_set,stop_lat,stop_lon,facility_stop_data)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        print(LS_TN_Radius_Original.shape)
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(LS_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_km_no_rule(filter_data_limit_seats,distance_max_set,stop_lat,stop_lon,facility_stop_data)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        print(LS_TN_Radius_Original.shape)
                else:
                    break
            
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val
            
            
            for index,row in vehicle_data_original_set.iterrows(): 
                seat_count = row.Seats
                vehicle_ID = row.ID
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_km_no_rule(filter_data_limit_seats,distance_max_set,stop_lat,stop_lon,facility_stop_data)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        print(GR_TN_Radius_Original.shape)
                    else:
                        filter_data_limit_seats = CalcDist.split_dataframe(GR_TN_Radius_Original, chunk_size=seat_count) 
                        #calculation
                        dataaset = CalcDist.generate_pick_km_no_rule(filter_data_limit_seats,distance_max_set,stop_lat,stop_lon,facility_stop_data)
                        index_values_assigning = dataaset[0]['All_Emp_Index']
                        data_for_trip = dataaset[0]['RouteEmployees']
                        data_for_trip_assign = pd.DataFrame(data_for_trip)
                        data_for_trip_assign['Vehicle_ID'] = vehicle_ID     
                        data_for_trip_assign['Vehicle_Trips'] = trip_assign_val
                        trip_assign_val = trip_assign_val + 1
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(data_for_trip_assign, ignore_index=True)
                        #drop the vehicle index from original sets
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        #drop the assigned employee from main set
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_assigning, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        print(GR_TN_Radius_Original.shape)
                else:
                    break
            
            
            
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_no_rule_km: '+ str(e) )
            test_logger.error('Failed to execute function generate_pick_no_rule_km : '+ str(e))
            return {"message": "internal server error, generate_pick_no_rule_km", "code": 501}, 501


    @classmethod
    # def generate_pick_with_rule(cls,GR_TN_Radius_Original,LS_TN_Radius_Original,vehicle_data_original_set,stop_lat,stop_lon,facility_stop_data): 
    def generate_pick_with_rule(cls,vehicle_data_original_set,Trip_Data_for_Drop,GR_TN_Radius_Original,LS_TN_Radius_Original,stop_lat,stop_lon,AssignEscortStatus,facility_stop_data): 
    
        try:
            columns_pick = ['ID', 'Name', 'LatLng', 
                                    'Gender', 'latitude',
            'longitude','DistanceToOffice','PeakSpeed',
            'Vehicle_ID','Vehicle_Trips','AssignEscort']
            
            assign_colsss = ['ID', 'Name', 'LatLng', 'Gender', 'latitude', 'longitude',
            'DistanceToOffice', 'PeakSpeed', 'Vehicle_ID', 'Vehicle_Trips',
            'AssignEscort', 'fac_latitude', 'fac_longitude', 'distance',
            'closest_stop_geom', 'KM_distance']
            
            Trip_Data_for_Drop = pd.DataFrame(columns = columns_pick) 
            
            trip_assign_val = 1
            for index,row in vehicle_data_original_set.iterrows(): 
                Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick) 
                seat_count = row.Seats
                vehicle_ID = row.ID
                
                if len(LS_TN_Radius_Original):
                    total_emp = LS_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                        
                        #assign pickorder
                        filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed'
                                        ]]
                        
                        
                        test_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                        long_dist_data = test_data[test_data['KM_distance'] == test_data['KM_distance'].max()].head(1)
                        
                        gender_type = long_dist_data.Gender.iloc[0]
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                        if gender_type == 'Female':
                            index_val = long_dist_data.index.values
                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_val, columns=None, level=None, inplace=True, errors='raise')
                            
                            if AssignEscortStatus:
                                filter_data_limit_seats['AssignEscort'] = 'Yes'
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                            
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        else:
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        
                        trip_assign_val = trip_assign_val + 1
                        #generate Route and Pick Order
                        loop_count = 1
                        for index,row in filter_data_limit_seats.iterrows():
                            if len(filter_data_limit_seats) > 0:
                                if loop_count == 1:
                                    filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                    one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                    maxlat = one_data.latitude.iloc[0]
                                    maxlon = one_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    
                                    index_pos = one_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                
                                    if len(filter_data_limit_seats) == 1:
                                        mlat = one_data.latitude.iloc[0]
                                        mlon = one_data.longitude.iloc[0]
                                        newer_data = mlat.astype(str)+","+mlon.astype(str)
                                        stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                        min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                        index_pos = min_filtered_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    else:
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        loop_count = loop_count + 1
                                else:
                                    filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                    min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                    maxlat = min_data.latitude.iloc[0]
                                    maxlon = min_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                    index_pos = min_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    loop_count = loop_count + 1
                            else:
                                loop_count = 1
                            
            
                        
                        #assign in order closest latlon
                        Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                        first_statusss_val = True
                        for index,row in Trip_Data_for_Drop_ordered.iterrows():
                            if first_statusss_val:
                                index__vall = index
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                first_statusss_val = False
                            else:
                                oldlat = row.latitude
                                oldlon = row.longitude
                                mergedlatlon = str(oldlat)+','+str(oldlon)
                                Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                index__vall = index
                        
                        #change latlon last value
                        last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                        getlastindex = last_data.index.values
                        fac_latlon = str(stop_lat)+','+str(stop_lon)
                        Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                        
                        #drop all the value from main set  
                        all_data_matched = LS_TN_Radius_Original.loc[LS_TN_Radius_Original['ID'].isin(Trip_Data_for_Drop_ordered.ID.values)]
                        #qdd to the dataframe
                        index_values_filtered = all_data_matched.index.values
                        LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                        
                        
                        #drop and resgenerate KM_distance
                        Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                        
                        #calculate the distance
                        lat = []
                        lon = []
                        for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                            # Try to,
                            try:
                                lat.append(row.split(',')[0])
                                lon.append(row.split(',')[1])
                            except:
                                lat.append(np.NaN)
                                lon.append(np.NaN)  
                    
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                        
                        
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                        #generate distance btwn and timebtn and timetooffice
                        Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                        #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                        
                        
                        #create a new derived column called TimeBtwEmp
                        timebtw = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                            
                        #create a new derived column called TimetoOffice
                        timetooffice = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                        
                        
                        #append it
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                        
                        
                    
                    else:
                        
                        if len(LS_TN_Radius_Original) == 1:
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            filter_data_limit_seats = LS_TN_Radius_Original.copy()
                            index_values_filtered = filter_data_limit_seats.index.values
                            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            
                            gender_type = filter_data_limit_seats.Gender.iloc[0]
                            if gender_type == 'Female':
                                if AssignEscortStatus:
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                else:
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                            
                            filter_data_limit_seats['closest_stop_geom'] = fac_latlon
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort','closest_stop_geom'
                                        ]]
                            lat = []
                            lon = []
                            for row in filter_data_limit_seats['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            filter_data_limit_seats['fac_latitude'] = lat
                            filter_data_limit_seats['fac_longitude'] = lon  
                            filter_data_limit_seats['fac_latitude'] = filter_data_limit_seats['fac_latitude'].astype(float)
                            filter_data_limit_seats['fac_longitude'] = filter_data_limit_seats['fac_longitude'].astype(float)
                            
                            for index,row in filter_data_limit_seats.iterrows():
                                filter_data_limit_seats.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                            #generate distance btwn and timebtn and timetooffice
                            filter_data_limit_seats['DistanceBTWEMP'] = filter_data_limit_seats['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        else:
                            
                            filter_data_limit_seats = LS_TN_Radius_Original.head(seat_count)
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed'
                                            ]]
                            
                            test_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                            long_dist_data = test_data[test_data['KM_distance'] == test_data['KM_distance'].max()].head(1)
                            
                            gender_type = long_dist_data.Gender.iloc[0]
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                            if gender_type == 'Female':
                                if AssignEscortStatus:
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                else:
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            
            
                            trip_assign_val = trip_assign_val + 1
                            
                            #qdd to the dataframe
                            index_values_filtered = filter_data_limit_seats.index.values
                            LS_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            LS_TN_Radius_Original = LS_TN_Radius_Original.reset_index(drop = True)
                            
                            #drop the unneccessary columns
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed',
                                            'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                            ]]
                            #generate Route and Pick Order
                            loop_count = 1
                            for index,row in filter_data_limit_seats.iterrows():
                                if len(filter_data_limit_seats) > 0:
                                    if loop_count == 1:
                                        filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                        one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                        maxlat = one_data.latitude.iloc[0]
                                        maxlon = one_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        
                                        index_pos = one_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    
                                        if len(filter_data_limit_seats) == 1:
                                            mlat = one_data.latitude.iloc[0]
                                            mlon = one_data.longitude.iloc[0]
                                            newer_data = mlat.astype(str)+","+mlon.astype(str)
                                            stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                            min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                            index_pos = min_filtered_data.index.values
                                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                            filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        else:
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            loop_count = loop_count + 1
                                    else:
                                        filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                        min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                        maxlat = min_data.latitude.iloc[0]
                                        maxlon = min_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                        index_pos = min_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        loop_count = loop_count + 1
                                else:
                                    loop_count = 1
                            
                            
                            #assign in order closest latlon
                            Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                            first_statusss_val = True
                            for index,row in Trip_Data_for_Drop_ordered.iterrows():
                                if first_statusss_val:
                                    index__vall = index
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    first_statusss_val = False
                                else:
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    mergedlatlon = str(oldlat)+','+str(oldlon)
                                    Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    index__vall = index
                            
                            #get the last data    
                            last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            
                            #drop and resgenerate KM_distance
                            Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                            
                            #calculate the distance
                            lat = []
                            lon = []
                            for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                            
                            
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                            
                            
                            #generate distance btwn and timebtn and timetooffice
                            Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                else:
                    break
                    
            
            if len(Trip_Data_for_Drop):
                trip_assign_val = Trip_Data_for_Drop.Vehicle_Trips.max()+1
            else:
                trip_assign_val = trip_assign_val
            
            
            for index,row in vehicle_data_original_set.iterrows(): 
                Trip_Data_for_Drop_ordered = pd.DataFrame(columns = columns_pick) 
                seat_count = row.Seats
                vehicle_ID = row.ID
                
                if len(GR_TN_Radius_Original):
                    total_emp = GR_TN_Radius_Original.shape[0]
                    if total_emp >= seat_count: 
                        filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                        
                        #assign pickorder
                        filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed'
                                        ]]
                        
                        
                        test_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                        long_dist_data = test_data[test_data['KM_distance'] == test_data['KM_distance'].max()].head(1)
                        
                        gender_type = long_dist_data.Gender.iloc[0]
                        vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                        if gender_type == 'Female':
                            index_val = long_dist_data.index.values
                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_val, columns=None, level=None, inplace=True, errors='raise')
                            
                            if AssignEscortStatus:
                                filter_data_limit_seats['AssignEscort'] = 'Yes'
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                            
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        else:
                            filter_data_limit_seats['AssignEscort'] = 'No'
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                        
                        trip_assign_val = trip_assign_val + 1
                        #generate Route and Pick Order
                        loop_count = 1
                        for index,row in filter_data_limit_seats.iterrows():
                            if len(filter_data_limit_seats) > 0:
                                if loop_count == 1:
                                    filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                    one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                    maxlat = one_data.latitude.iloc[0]
                                    maxlon = one_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    
                                    index_pos = one_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                
                                    if len(filter_data_limit_seats) == 1:
                                        mlat = one_data.latitude.iloc[0]
                                        mlon = one_data.longitude.iloc[0]
                                        newer_data = mlat.astype(str)+","+mlon.astype(str)
                                        stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                        min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                        index_pos = min_filtered_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    else:
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                        loop_count = loop_count + 1
                                else:
                                    filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                    min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                    maxlat = min_data.latitude.iloc[0]
                                    maxlon = min_data.longitude.iloc[0]
                                    stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                    Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                    index_pos = min_data.index.values
                                    filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                    filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    loop_count = loop_count + 1
                            else:
                                loop_count = 1
                            
            
                        
                        #assign in order closest latlon
                        Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                        first_statusss_val = True
                        for index,row in Trip_Data_for_Drop_ordered.iterrows():
                            if first_statusss_val:
                                index__vall = index
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                first_statusss_val = False
                            else:
                                oldlat = row.latitude
                                oldlon = row.longitude
                                mergedlatlon = str(oldlat)+','+str(oldlon)
                                Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                index__vall = index
                        
                        #change latlon last value
                        last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                        getlastindex = last_data.index.values
                        fac_latlon = str(stop_lat)+','+str(stop_lon)
                        Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                        
                        #drop all the value from main set  
                        all_data_matched = GR_TN_Radius_Original.loc[GR_TN_Radius_Original['ID'].isin(Trip_Data_for_Drop_ordered.ID.values)]
                        #qdd to the dataframe
                        index_values_filtered = all_data_matched.index.values
                        GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                        GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                        
                        
                        #drop and resgenerate KM_distance
                        Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                        
                        #calculate the distance
                        lat = []
                        lon = []
                        for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                            # Try to,
                            try:
                                lat.append(row.split(',')[0])
                                lon.append(row.split(',')[1])
                            except:
                                lat.append(np.NaN)
                                lon.append(np.NaN)  
                    
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                        Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                        Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                        
                        
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                        #generate distance btwn and timebtn and timetooffice
                        Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                        #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                        
                        
                        #create a new derived column called TimeBtwEmp
                        timebtw = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                            
                        #create a new derived column called TimetoOffice
                        timetooffice = 0
                        for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                            timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                            Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                        
                        
                        #append it
                        Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                        
                        
                    
                    else:
                        
                        if len(GR_TN_Radius_Original) == 1:
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            filter_data_limit_seats = GR_TN_Radius_Original.copy()
                            index_values_filtered = filter_data_limit_seats.index.values
                            GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                            filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                            filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            
                            gender_type = filter_data_limit_seats.Gender.iloc[0]
                            if gender_type == 'Female':
                                if AssignEscortStatus:
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                else:
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                            
                            filter_data_limit_seats['closest_stop_geom'] = fac_latlon
                            trip_assign_val = trip_assign_val + 1
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                            
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                'Gender', 'latitude',
                                        'longitude','DistanceToOffice','PeakSpeed',
                                        'Vehicle_ID','Vehicle_Trips','AssignEscort','closest_stop_geom'
                                        ]]
                            lat = []
                            lon = []
                            for row in filter_data_limit_seats['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            filter_data_limit_seats['fac_latitude'] = lat
                            filter_data_limit_seats['fac_longitude'] = lon  
                            filter_data_limit_seats['fac_latitude'] = filter_data_limit_seats['fac_latitude'].astype(float)
                            filter_data_limit_seats['fac_longitude'] = filter_data_limit_seats['fac_longitude'].astype(float)
                            
                            for index,row in filter_data_limit_seats.iterrows():
                                filter_data_limit_seats.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                        
                        
                            #generate distance btwn and timebtn and timetooffice
                            filter_data_limit_seats['DistanceBTWEMP'] = filter_data_limit_seats['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in filter_data_limit_seats.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                filter_data_limit_seats.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(filter_data_limit_seats, ignore_index=True)
                        else:
                            
                            filter_data_limit_seats = GR_TN_Radius_Original.head(seat_count)
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed'
                                            ]]
                            
                            test_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                            long_dist_data = test_data[test_data['KM_distance'] == test_data['KM_distance'].max()].head(1)
                            
                            gender_type = long_dist_data.Gender.iloc[0]
                            vehicle_data_original_set.drop(labels=None, axis=0, index=index, columns=None, level=None, inplace=True, errors='raise')
                        
                        
                            if gender_type == 'Female':
                                if AssignEscortStatus:
                                    filter_data_limit_seats['AssignEscort'] = 'Yes'
                                else:
                                    filter_data_limit_seats['AssignEscort'] = 'No'
                                    
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            else:
                                filter_data_limit_seats['AssignEscort'] = 'No'
                                filter_data_limit_seats['Vehicle_ID'] = vehicle_ID     
                                filter_data_limit_seats['Vehicle_Trips'] = trip_assign_val
                            
            
                            trip_assign_val = trip_assign_val + 1
                            
                            #qdd to the dataframe
                            index_values_filtered = filter_data_limit_seats.index.values
                            GR_TN_Radius_Original.drop(labels=None, axis=0, index=index_values_filtered, columns=None, level=None, inplace=True, errors='raise')
                            GR_TN_Radius_Original = GR_TN_Radius_Original.reset_index(drop = True)
                            
                            #drop the unneccessary columns
                            
                            #assign pickorder
                            filter_data_limit_seats = filter_data_limit_seats[['ID', 'Name', 'LatLng', 
                                                                    'Gender', 'latitude',
                                            'longitude','DistanceToOffice','PeakSpeed',
                                            'Vehicle_ID','Vehicle_Trips','AssignEscort'
                                            ]]
                            #generate Route and Pick Order
                            loop_count = 1
                            for index,row in filter_data_limit_seats.iterrows():
                                if len(filter_data_limit_seats) > 0:
                                    if loop_count == 1:
                                        filtered_data = CalcDist.distance_data(facility_stop_data,filter_data_limit_seats)
                                        one_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()].head(1)
                                        maxlat = one_data.latitude.iloc[0]
                                        maxlon = one_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        
                                        index_pos = one_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                    
                                        if len(filter_data_limit_seats) == 1:
                                            mlat = one_data.latitude.iloc[0]
                                            mlon = one_data.longitude.iloc[0]
                                            newer_data = mlat.astype(str)+","+mlon.astype(str)
                                            stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                            min_filtered_data = CalcDist.distance_data(stop_data_near,filter_data_limit_seats)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_filtered_data, ignore_index=True)
                                            index_pos = min_filtered_data.index.values
                                            filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                            filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        else:
                                            Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(one_data, ignore_index=True)
                                            loop_count = loop_count + 1
                                    else:
                                        filtered_data = CalcDist.distance_data(stopss_copy,filter_data_limit_seats)
                                        min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()].head(1)
                                        maxlat = min_data.latitude.iloc[0]
                                        maxlon = min_data.longitude.iloc[0]
                                        stopss_copy = CalcDist.create_stop_df(maxlat,maxlon)
                                        Trip_Data_for_Drop_ordered = Trip_Data_for_Drop_ordered.append(min_data, ignore_index=True)
                                        index_pos = min_data.index.values
                                        filter_data_limit_seats.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                                        filter_data_limit_seats = filter_data_limit_seats.reset_index(drop = True)
                                        loop_count = loop_count + 1
                                else:
                                    loop_count = 1
                            
                            
                            #assign in order closest latlon
                            Trip_Data_for_Drop_ordered_new = pd.DataFrame(columns=assign_colsss)
                            first_statusss_val = True
                            for index,row in Trip_Data_for_Drop_ordered.iterrows():
                                if first_statusss_val:
                                    index__vall = index
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    first_statusss_val = False
                                else:
                                    oldlat = row.latitude
                                    oldlon = row.longitude
                                    mergedlatlon = str(oldlat)+','+str(oldlon)
                                    Trip_Data_for_Drop_ordered_new.at[index__vall,'closest_stop_geom'] = mergedlatlon
                                    Trip_Data_for_Drop_ordered_new = Trip_Data_for_Drop_ordered_new.append(row, ignore_index=True)
                                    index__vall = index
                            
                            #get the last data    
                            last_data = Trip_Data_for_Drop_ordered_new.tail(1)
                            getlastindex = last_data.index.values
                            fac_latlon = str(stop_lat)+','+str(stop_lon)
                            Trip_Data_for_Drop_ordered_new.at[getlastindex,'closest_stop_geom'] = fac_latlon
                            
                            #drop and resgenerate KM_distance
                            Trip_Data_for_Drop_ordered_new.drop(['KM_distance', 'fac_latitude', 'fac_longitude', 'distance'], axis=1, inplace=True)    
                            
                            #calculate the distance
                            lat = []
                            lon = []
                            for row in Trip_Data_for_Drop_ordered_new['closest_stop_geom']:
                                # Try to,
                                try:
                                    lat.append(row.split(',')[0])
                                    lon.append(row.split(',')[1])
                                except:
                                    lat.append(np.NaN)
                                    lon.append(np.NaN)  
                        
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = lat
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = lon  
                            Trip_Data_for_Drop_ordered_new['fac_latitude'] = Trip_Data_for_Drop_ordered_new['fac_latitude'].astype(float)
                            Trip_Data_for_Drop_ordered_new['fac_longitude'] = Trip_Data_for_Drop_ordered_new['fac_longitude'].astype(float)
                            
                            
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                Trip_Data_for_Drop_ordered_new.at[index,'KM_distance'] = CalcDist.distance_two_points(row.longitude, row.latitude, row.fac_longitude, row.fac_latitude)
                            
                            
                            #generate distance btwn and timebtn and timetooffice
                            Trip_Data_for_Drop_ordered_new['DistanceBTWEMP'] = Trip_Data_for_Drop_ordered_new['KM_distance']
                            #Trip_Data_for_Drop_ordered_new_val = DistanceBTWEMP_Calculate(Trip_Data_for_Drop_ordered_new,stop_lon,stop_lat)
                            
                            
                            #create a new derived column called TimeBtwEmp
                            timebtw = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timebtw = (row.DistanceBTWEMP*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimeBtwEmp'] = timebtw
                                
                            #create a new derived column called TimetoOffice
                            timetooffice = 0
                            for index,row in Trip_Data_for_Drop_ordered_new.iterrows():
                                timetooffice = (row.DistanceToOffice*60)/row.PeakSpeed
                                Trip_Data_for_Drop_ordered_new.at[index,'TimetoOffice'] = timetooffice 
                            
                            
                            #append it
                            Trip_Data_for_Drop = Trip_Data_for_Drop.append(Trip_Data_for_Drop_ordered_new, ignore_index=True)
                else:
                    break
            
            
            return Trip_Data_for_Drop    
        except Exception as e:
            logging.error('Failed to execute function generate_pick_with_rule: '+ str(e) )
            return {"message": "internal server error, generate_pick_with_rule", "code": 501}, 501






    @classmethod
    def calc_distance(cls,data_set_hundred_filtered_loop,km,loop_count,occur_val,facility_stop_data,linestring_df,cluster):
        try:
            
            km = 0
            loop_count = 1   
            stopss_copy = facility_stop_data.copy()
            if len(data_set_hundred_filtered_loop) == 1:
                filtered_data = CalcDist.distance_data(stopss_copy,data_set_hundred_filtered_loop)
                filtered_data['Route'] = occur_val
                #check clustering done
                filtered_data['Is_Clustered'] = 'Yes'
                linestring_df = linestring_df.append(filtered_data, ignore_index=True)
                occur_val = occur_val + 1
            elif len(data_set_hundred_filtered_loop) == 2:
                km_val = 0                         
                filtered_data = CalcDist.distance_data(stopss_copy,data_set_hundred_filtered_loop)
                max_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].max()]
                min_data = filtered_data[filtered_data['KM_distance'] == filtered_data['KM_distance'].min()]
                min_data_one = min_data.copy()

                mlat = max_data.latitude.iloc[0]
                mlon = max_data.longitude.iloc[0]
                newer_data = mlat.astype(str)+","+mlon.astype(str)

                #add the minimum data into the route
                stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                min_data.drop(['fac_latitude','fac_longitude', 'distance', 'closest_stop_geom', 'KM_distance'], axis=1, inplace=True)
                
                min_filtered_data = CalcDist.distance_data(stop_data_near,min_data)
                
                km_value = min_filtered_data.KM_distance.iloc[0]
                km_val = km_val + km_value

                if km_val < MaxEmplPickupRadius:
                    min_filtered_data['Route'] = occur_val
                    #check clustering done
                    min_filtered_data['Is_Clustered'] = 'No'
                    linestring_df = linestring_df.append(min_filtered_data, ignore_index=True)
                    max_data['Route'] = occur_val
                    #check clustering done
                    max_data['Is_Clustered'] = 'No'
                    linestring_df = linestring_df.append(max_data, ignore_index=True)
                    occur_val = occur_val + 1
                else:
                    min_data_one['Route'] = occur_val
                    #check clustering done
                    min_data_one['Is_Clustered'] = 'No'
                    linestring_df = linestring_df.append(min_data_one, ignore_index=True)
                    occur_val = occur_val + 1
                    max_data['Route'] = occur_val
                    max_data['Is_Clustered'] = 'No'
                    linestring_df = linestring_df.append(max_data, ignore_index=True)
                    occur_val = occur_val + 1    
            else:
                
                for index,row in data_set_hundred_filtered_loop.iterrows():

                    if len(data_set_hundred_filtered_loop) > 0:

                        if loop_count == 1:
                            filtered_data = CalcDist.distance_data(stopss_copy,data_set_hundred_filtered_loop)
                            max_result = CalcDist.find_max(filtered_data)

                            stopss_copy = max_result[1]
                            one_data = max_result[0]

                            index_pos = one_data.index.values
                            data_set_hundred_filtered_loop.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                            data_set_hundred_filtered_loop = data_set_hundred_filtered_loop.reset_index(drop = True)

                            if len(data_set_hundred_filtered_loop) == 1:
                                
                                km_val = 0 
                                mlat = one_data.latitude.iloc[0]
                                mlon = one_data.longitude.iloc[0]
                                newer_data = mlat.astype(str)+","+mlon.astype(str)
                                stop_data_near = CalcDist.create_stop_df(mlat,mlon)
                                min_filtered_data = CalcDist.distance_data(stop_data_near,data_set_hundred_filtered_loop)
                                km_value = min_filtered_data.KM_distance.iloc[0]
                                km_val = km_val + km_value

                                rem_data = one_data.copy()
                                
                                if km_val < MaxEmplPickupRadius:
                                    min_filtered_data['Route'] = occur_val
                                    #check clustering done
                                    min_filtered_data['Is_Clustered'] = 'Yess'
                                    linestring_df = linestring_df.append(min_filtered_data, ignore_index=True)
                                    rem_data['Route'] = occur_val
                                    #check clustering done
                                    rem_data['Is_Clustered'] = 'Yes'
                                    linestring_df = linestring_df.append(rem_data, ignore_index=True)
                                    occur_val = occur_val + 1
                                else:
                                    min_filtered_data['Route'] = occur_val
                                    #check clustering done
                                    min_filtered_data['Is_Clustered'] = 'Yes1'
                                    linestring_df = linestring_df.append(min_filtered_data, ignore_index=True)
                                    occur_val = occur_val + 1
                                    rem_data['Route'] = occur_val
                                    rem_data['Is_Clustered'] = 'Yes'
                                    linestring_df = linestring_df.append(rem_data, ignore_index=True)
                                    occur_val = occur_val + 1 

                            elif len(data_set_hundred_filtered_loop) < 1:
                                emp_id = one_data['ID'].iloc[0]
                                if emp_id not in linestring_df.values:    
                                    ssslat = str(facility_stop_data.fac_latitude.iloc[0])
                                    ssslon = str(facility_stop_data.fac_longitude.iloc[0])
                                    new_data = ssslat+","+ssslon
                                    one_data['closest_stop_geom'] = new_data
                                    one_data['Route'] = occur_val
                                    one_data['Is_Clustered'] = 'Yes'
                                    linestring_df = linestring_df.append(one_data, ignore_index=True)

                                    loop_count = 1
                                    occur_val = occur_val + 1

                            else:

                                loop_count = loop_count + 1
                        else: 
                            
                            filtered_data = CalcDist.distance_data(stopss_copy,data_set_hundred_filtered_loop)
                            max_result = CalcDist.find_min(filtered_data)

                            stopss_copy = max_result[1]
                            min_data = max_result[0] 

                            index_pos = min_data.index.values
                            data_set_hundred_filtered_loop.drop(labels=None, axis=0, index=index_pos, columns=None, level=None, inplace=True, errors='raise')
                            data_set_hundred_filtered_loop = data_set_hundred_filtered_loop.reset_index(drop = True)

                            ssslat = min_data.latitude.iloc[0]
                            ssslon = min_data.longitude.iloc[0]
                            new_data = ssslat.astype(str)+","+ssslon.astype(str)
                            one_data['closest_stop_geom'] = new_data
                            one_data['Route'] = occur_val
                            one_data['Is_Clustered'] = 'Yes'
                            linestring_df = linestring_df.append(one_data, ignore_index=True)

                            one_data = min_data.copy()
                            val = one_data.KM_distance.iloc[0]
                            km = km+val


                            if km > MaxEmplPickupRadius:
                                ssslat = str(facility_stop_data.fac_latitude.iloc[0])
                                ssslon = str(facility_stop_data.fac_longitude.iloc[0])
                                new_data = ssslat+","+ssslon
                                one_data['closest_stop_geom'] = new_data
                                one_data['Route'] = occur_val
                                one_data['Is_Clustered'] = 'Yes'
                                linestring_df = linestring_df.append(one_data, ignore_index=True)
                                stopss_copy = facility_stop_data.copy()
                                loop_count = 1
                                occur_val = occur_val + 1
                                km = 0  
                            else:
                                loop_count = loop_count + 1

                            if len(data_set_hundred_filtered_loop) < 1 and km < MaxEmplPickupRadius:
                                emp_id = one_data['ID'].iloc[0]
                                if emp_id not in linestring_df.values:
                                    ssslat = str(facility_stop_data.fac_latitude.iloc[0])
                                    ssslon = str(facility_stop_data.fac_longitude.iloc[0])
                                    new_data = ssslat+","+ssslon
                                    one_data['closest_stop_geom'] = new_data
                                    one_data['Route'] = linestring_df.Route.max()
                                    one_data['Is_Clustered'] = 'Yes'
                                    linestring_df = linestring_df.append(one_data, ignore_index=True)
                                    stopss_copy = facility_stop_data.copy()
                                    loop_count = 1
                                    occur_val = occur_val + 1
                    else:
                        loop_count = 1
                        
            return linestring_df
        except Exception as e:
            logging.error('Failed to execute function calc_distance: '+ str(e) )
            test_logger.error('Failed to execute function calc_distance : '+ str(e))
            return {"message": "internal server error, function calc_dist failed to execute", "code": 501}, 501

        















