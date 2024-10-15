import pandas as pd
import logging
import logstash
import sys
 
host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))


class CreateStopDF:
    ''' Implementing Distance Calculation from 2 points '''

    @classmethod
    def create_stop_df(cls, stop_lat,stop_lon):
        try:
            stop_data = pd.DataFrame(columns=['fac_latitude', 'fac_longitude'])
            stop_data.at[0,'fac_latitude'] = stop_lat
            stop_data.at[0,'fac_longitude'] = stop_lon
            return stop_data
        except Exception as e:
            logging.error('Failed to execute function CreateStopDF: '+ str(e) )
            test_logger.error('Failed to execute function CreateStopDF : '+ str(e))
            return {"message": "internal server error", "code": 501}, 501
