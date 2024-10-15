import pandas as pd
import numpy as np

import logging
import logstash
import sys
 
host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class CalculateDistanceFromTwoPoints:
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
            logging.error('Failed to execute function haversine_np: '+ str(e) )
            test_logger.error('Failed to execute function haversine_np : '+ str(e))
            return {"message": "internal server error, haversine_np", "code": 501}, 501

    @classmethod
    def distance_data(cls,stopss_copy,data_set_hundred_filtered_loop):
        try:
            stop_lon = stopss_copy['fac_longitude'].iloc[0]
            stop_lat = stopss_copy['fac_latitude'].iloc[0]
            df = pd.DataFrame(data={'lon1':data_set_hundred_filtered_loop['longitude'],'lon2':stop_lon,'lat1':data_set_hundred_filtered_loop['latitude'],'lat2':stop_lat})
            closest_stops = CalculateDistanceFromTwoPoints.haversine_np(df['lon1'],df['lat1'],df['lon2'],df['lat2'])
            closest_stops['closest_stop_geom'] = [','.join(str(x) for x in y) for y in map(tuple, closest_stops[['fac_latitude', 'fac_longitude']].values)]
            closest_stops['KM_distance'] = closest_stops['distance'].apply(lambda x: x/1000)
            data_set_copy1 = data_set_hundred_filtered_loop.join(closest_stops)
            return data_set_copy1
        except Exception as e:
            logging.error('Failed to execute function distance_data: '+ str(e) )
            test_logger.error('Failed to execute function distance_data : '+ str(e))
            return {"message": "internal server error, distance_data", "code": 501}, 501




