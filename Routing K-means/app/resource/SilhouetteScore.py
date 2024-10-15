from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering

import logging
import logstash
import sys
 
host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class SilhouetteScoreCalc:
    ''' Calculate the Silhouette Score '''

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






