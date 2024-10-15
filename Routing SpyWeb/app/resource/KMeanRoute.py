from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd


import logging
import logstash
import sys
 
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import plotly.express as px

host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

# @Madhu
# Elbow Method to find optimum cluster count

class KMeansGroupError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code
        super().__init__(self.message)


class KMeanRoute:

    """
    @staticmethod
    def cluster_scatter_plot(X_std, cluster_labels, cluster_centers, n_cluster):
    
        # Scatter plot for clusters
        fig = go.Figure()
        # Add cluster points
        fig.add_trace(go.Scatter(x=X_std[:, 0], y=X_std[:, 1], mode='markers', marker=dict(
                color=cluster_labels,
                colorscale='Viridis',
                showscale=True
            ),
            name='Clusters'
        ))

        # Add centroids
        fig.add_trace(go.Scatter(x=cluster_centers[:, 0], y=cluster_centers[:, 1], mode='markers', marker=dict(   # x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1]
                size=12,
                color='red',
                symbol='x'
            ),
            name='Centroids'
        ))

        # Set plot title and axis labels
        fig.update_layout(
            title=f'Cluster Visualization for {n_cluster} clusters',
            xaxis_title='Standardized Latitude',
            yaxis_title='Standardized Longitude',
            showlegend=True
        )

        return fig
    
    @staticmethod
    def cluster_bar_plot(unique_labels, counts, n_cluster):

        # Bar plot for cluster counts
        bar_fig = px.bar(
            x=unique_labels,
            y=counts,
            labels={'x': 'Cluster Labels', 'y': 'Number of Data Points'},
            title=f'Number of Data Points in Each Cluster (n_clusters={n_cluster})',
        )

        # Customize the layout
        bar_fig.update_layout(xaxis=dict(tickmode='array', tickvals=unique_labels), showlegend=False)

        return bar_fig
    
    @staticmethod
    def plot_elbow(cluster_range, wcss, optimal_index):

        # Plot WCSS
        plt.plot(cluster_range, wcss, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')
        plt.axvline(x=optimal_index, color='r', linestyle='--', label=f'Optimal: {optimal_index}')
        plt.legend()

        return plt
"""

    @staticmethod
    def get_optimal_index_elbow(cluster_range, X_std_d, thr):
        
        try:

            wcss = []
            for n_cluster in cluster_range:
                kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=500, random_state=42, algorithm='auto')
                cluster_labels = kmeans.fit_predict(X_std_d)
                wcss.append(kmeans.inertia_)

                unique_labels, counts = np.unique(cluster_labels, return_counts=True)

                # # Plot clusters
                # cluster_centers = kmeans.cluster_centers_
                # sca_fig = cluster_scatter_plot(X_std, cluster_labels, cluster_centers, n_cluster)
                # sca_fig.show()

                # Plot unique lable counts
                # bar_fig = cluster_bar_plot(unique_labels, counts, n_cluster)
                # bar_fig.show()

            # Calculate differences
            if len(wcss) < 3:  # Ensure there are enough elements to calculate second diff
                return 2  # Fallback to the smallest cluster number

            wcss_diff = np.diff(wcss)
            wcss_diff2 = np.diff(wcss_diff)

            if len(wcss_diff2) == 0:
                return cluster_range[0]  # Fallback to the smallest cluster number

            threshold = np.max(wcss_diff2) * thr  # Example threshold, adjust as needed
            indices = np.where(wcss_diff2 > threshold)[0]
            if indices.size > 0:
                optimal_index = indices[-1] + 2 # Adjust index due to np.diff reduction
            else:
                optimal_index = cluster_range[np.argmax(wcss_diff2) + 2]

            # # Plow elbow
            # elbow_plot = plot_elbow(cluster_range, wcss, optimal_index)
            # elbow_plot.show()

            return optimal_index #, sca_fig, bar_fig, elbow_plot
        except Exception as e:

            logging.error('Failed to execute function get_optimal_index_elbow: '+ str(e) )
            return {"message": "internal server error", "code": 501}
        

    @staticmethod
    def k_means_group(df, limit_size, thr):

        try:
                
            std_df = StandardScaler().fit_transform(df[['latitude','longitude']])
            if(len(std_df) < 50):
                limit_size = len(std_df)
            cluster_range = range(2, limit_size)  # Start from 2 because silhouette score is not defined for 1 cluster

            optimal_clusters = KMeanRoute.get_optimal_index_elbow(cluster_range, std_df, thr)

            kmeans_model = KMeans(n_clusters= optimal_clusters, init='k-means++', max_iter=500, random_state=42, algorithm='auto')
            cluster_labels = kmeans_model.fit_predict(std_df) 
            cluster_centroids = kmeans_model.cluster_centers_

            # Convert cluster centroids back to original scale
            scaler = StandardScaler().fit(df[['latitude', 'longitude']])
            original_centroids = scaler.inverse_transform(cluster_centroids)

            unique_labels, inverse_indices = np.unique(cluster_labels, return_inverse=True)
            labels_with_counts = np.column_stack((unique_labels, np.bincount(inverse_indices)))
            labels_with_counts_list = labels_with_counts.tolist()
            labels_with_counts_list.sort(key = lambda labels_with_counts_list: labels_with_counts_list[1])

            df['groupNumber'] = cluster_labels
            sorted_df = df.sort_values(by='groupNumber')

            return sorted_df
        except Exception as e:

            logging.error('Failed to execute function k_means_group: '+ str(e) )
            raise KMeansGroupError("Internal Server Error", 501)
