import os
# from datetime import datetime
import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
import numpy as np
import uuid
import json

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

from resource.Createstopdataframe import CreateStopDF
from resource.SplitLocation import SplitEmployeeLocation
from resource.DistanceCalculation import CalculateDistanceFromTwoPoints
from resource.SilhouetteScore import SilhouetteScoreCalc
from resource.CalcDistanceMain import CalcDist

from resource.CityCircle import CityCircle
from resource.RoutePlan import RoutePlan
from resource.RouteResponse import RouteResponse
from resource.KMeanRoute import KMeanRoute

import KMeans_Method
import logging
import logstash
import sys
 
from geopy.distance import geodesic
import math


host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

PORT = 8000
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)



def set_single_employee_details(df, center):
    """Handle case where there is only one employee."""
    try:
        df[['PickOrder', 'DistanceToNextEmployee', 'closest_stop_Latitude', 'closest_stop_Longitude']] = None
        lat, lon = SplitEmployeeLocation.split_lat_lon(df)
        df['latitude'] = list(map(float, lat))
        df['longitude'] = list(map(float, lon))

        row = df.iloc[0]
        row_index = row.name
        row_latlng = (row['latitude'], row['longitude'])

        df.at[row_index, 'PickOrder'] = 1
        df.at[row_index, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance(row_latlng, center)
        df.at[row_index, 'closest_stop_Latitude'] = center[0]
        df.at[row_index, 'closest_stop_Longitude'] = center[1]
        df.at[row_index, 'RouteCode'] = str(uuid.uuid4())

        selected_columns = [
            'LatLng', 'RouteCode', 'PickOrder', 'distance', 'DistanceToNextEmployee', 'closest_stop_Latitude', 'closest_stop_Longitude'
        ]
        
        return RouteResponse.get_response_df(df, selected_columns)
    except Exception as e:
        logging.error('Failed to execute function set_single_employee_details: '+ str(e))
        raise
    


def set_two_employee_details(df, center, max_emp_pickup_arc):
    """Handle case where there are exactly two employees."""
    try:
        df[['PickOrder', 'DistanceToNextEmployee', 'closest_stop_Latitude', 'closest_stop_Longitude']] = None
        
        lat, lon = SplitEmployeeLocation.split_lat_lon(df)
        df['latitude'] = list(map(float, lat))
        df['longitude'] = list(map(float, lon))
        
        row1, row2 = df.iloc[0], df.iloc[1]
        row1_latlng, row2_latlng = (row1['latitude'], row1['longitude']), (row2['latitude'], row2['longitude'])
        
        dist = RoutePlan.calculate_distance(row1_latlng, row2_latlng)
        
        if dist > max_emp_pickup_arc:
            for row, idx in zip([row1, row2], [row1.name, row2.name]):
                df.at[idx, 'PickOrder'] = 1
                df.at[idx, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance((row['latitude'], row['longitude']), center)
                df.at[idx, 'closest_stop_Latitude'] = center[0]
                df.at[idx, 'closest_stop_Longitude'] = center[1]
                df.at[idx, 'RouteCode'] = str(uuid.uuid4())
        else:
            col = "distance"
            leader, follower = (row1, row2) if row1[col] > row2[col] else (row2, row1)
            leader_idx, follower_idx = leader.name, follower.name
            
            df.at[leader_idx, 'PickOrder'] = 1
            df.at[leader_idx, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance((leader['latitude'], leader['longitude']), (follower['latitude'], follower['longitude']))
            df.at[leader_idx, 'closest_stop_Latitude'] = follower['latitude']
            df.at[leader_idx, 'closest_stop_Longitude'] = follower['longitude']
            
            df.at[follower_idx, 'PickOrder'] = 2
            df.at[follower_idx, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance((follower['latitude'], follower['longitude']), center)
            df.at[follower_idx, 'closest_stop_Latitude'] = center[0]
            df.at[follower_idx, 'closest_stop_Longitude'] = center[1]
            
            route_code = str(uuid.uuid4())
            df.loc[[leader_idx, follower_idx], 'RouteCode'] = route_code

        selected_columns = [
            'LatLng', 'RouteCode', 'PickOrder', 'distance', 'DistanceToNextEmployee', 'closest_stop_Latitude', 'closest_stop_Longitude'
        ]
        
        return RouteResponse.get_response_df(df, selected_columns)
    except Exception as e:
        logging.error('Failed to execute function set_two_employee_details: '+ str(e))
        raise

def set_multiple_employee_details(df, center, limit_size, thr):
    """Handle case where there are more than two employees."""
    try:
        lat, lon = SplitEmployeeLocation.split_lat_lon(df)
        df['latitude'] = list(map(float, lat))
        df['longitude'] = list(map(float, lon))
        
        grouped_df = KMeanRoute.k_means_group(df, limit_size, thr)
        pick_order_df = RoutePlan.assign_next_point(grouped_df, center)
        final_df = RoutePlan.generate_unique_route_codes(pick_order_df, 'groupNumber')

        selected_columns = [
            'MemberID', 'LatLng', 'RouteCode', 'PickOrder', 'distance', 'DistanceToNextEmployee', 'closest_stop_Latitude', 'closest_stop_Longitude'
        ]
        
        return RouteResponse.get_response_df(final_df, selected_columns)
    except Exception as e:
        logging.error('Failed to execute function set_multiple_employee_details: '+ str(e))
        raise

def process_employee_df(employee_df, center, max_emp_pickup_arc, limit_size, thr):
    """Process employee DataFrame and return cleaned and organized DataFrame."""
    
    if employee_df.empty:
        return pd.DataFrame()
    
    try:
        n_employees = len(employee_df)
        if n_employees == 1:
            return set_single_employee_details(employee_df, center)
        elif n_employees == 2:
            return set_two_employee_details(employee_df, center, max_emp_pickup_arc)
        else:
            return set_multiple_employee_details(employee_df, center, limit_size, thr)
    except Exception as e:
        logging.error('Failed to execute function process_employee_df: '+ str(e))
        raise


class InitRoutePlannerAPI(Resource):

    def __init__(self):
        self.now = datetime.datetime.now()
        self.curr_time = self.now.strftime("%H.%M.%S")
        self.MYDIR = self.now.strftime("%Y.%m.%d")

    def post(self):
        print('Loading.....')

        try:
            #Getting Json Input
            return_data = request.get_json() 
            
            limit_size = 50
            thr = 0.01

            max_emp_pickup_arc = return_data['MaxEmplPickupRadius'] * 2
            if return_data['MaxEmplPickupRadius'] is None or return_data['MaxEmplPickupRadius'] <= 0:
                raise ValueError("MaxEmplPickupRadius must be a valid positive number.")

            # Concentric circle radius
            max_emp_pickup_radius = return_data['EmplSubGroupRadius']
            if max_emp_pickup_radius is None or max_emp_pickup_radius <= 0:
                raise ValueError("EmplSubGroupRadius must be a valid positive number.")


            is_manual_distance = return_data['IsManualDistance']
            if isinstance(is_manual_distance, str):
                if is_manual_distance.lower() == 'true':
                    is_manual_distance = True
                elif is_manual_distance.lower() == 'false':
                    is_manual_distance = False
                else:
                    raise ValueError("IsManualDistance must be a boolean value (True or False).")
            elif not isinstance(is_manual_distance, bool):
                raise ValueError("IsManualDistance must be a boolean value.")
            
            # city Radius/boundary
            city_radius = return_data['ClusteringRadius']
            if city_radius is None or city_radius <= 0:
                raise ValueError("ClusteringRadius must be a valid positive number.")
            
            #Facility Lat&Lon
            latlong = return_data['FacilityLatLong']
            center = tuple(map(float, latlong.strip().split(',')))
            latitude, longitude = center
            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude must be between -90 and 90 degrees.")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude must be between -180 and 180 degrees.")

            #Employee Details and Convert to dataframe
            emp_data = return_data['Members']
            employee = pd.DataFrame(emp_data)
            if employee['MemberID'].isnull().any():
                count_missing = employee['MemberID'].isnull().sum()
                raise ValueError(f"{count_missing} MemberIDs have null values in 'MemberID'.")

            invalid_members = []

            for index, row in employee.iterrows():
                latlng = row['LatLng']
                member_id = row['MemberID']

                latitude, longitude = map(float, latlng.strip().split(','))
                if not (-90 <= latitude <= 90):
                    invalid_members.append(member_id)
                if not (-180 <= longitude <= 180):
                    invalid_members.append(member_id)
                if invalid_members:
                    raise ValueError(f"The following MemberIDs invalid LatLng: {', '.join(invalid_members)}.")
                

            if is_manual_distance:
                employee = employee[['MemberID', 'LatLng', 'ManualDistance']].rename(columns={'ManualDistance': 'distance'})
            else:
                employee = employee[['MemberID', 'LatLng', 'DistanceToOffice']].rename(columns={'DistanceToOffice': 'distance'})

            null_manual_distances = employee[employee['distance'].isnull()]['MemberID']
            if not null_manual_distances.empty:
                raise ValueError(f"The following MemberIDs have null values in distance: {', '.join(null_manual_distances)}.")
            
            in_city_emp_df = employee[employee['distance'] <= city_radius]
            out_city_emp_df = employee[employee['distance'] > city_radius]

            if not in_city_emp_df.empty:
                # Get number of initial route to consider
                num_routes = CityCircle.split_circle_by_arc(city_radius, max_emp_pickup_arc)

                # Get starting point of initial route considered
                route_init_latlng = CityCircle.get_intersection_coordinates(center, city_radius, num_routes)

                # split into sectors/Create groups based on distance of each employee to each route  
                in_city_emp_sectored_df = CityCircle.find_closest_line_to_point(center, in_city_emp_df, route_init_latlng)

                # Split the sectors/groups to smaller bins 
                in_city_emp_bin_df = CityCircle.apply_split_to_routes(in_city_emp_sectored_df, city_radius, max_emp_pickup_radius)

                # Set employee pick up order from farthest bin to closest bin in each route
                in_city_emp_pick_order_df = RoutePlan.set_pick_order(in_city_emp_bin_df)
                
                # calc distance to next employee in route, get co-ordinates to next employee on the route
                in_city_emp_route_dist_df = RoutePlan.get_distance_to_next_emp(in_city_emp_pick_order_df, center)
                
                # Added unique route code to df ### Used in Adding new employee to the group
                in_city_emp_final_df = RoutePlan.generate_unique_route_codes(in_city_emp_route_dist_df, "route")

                # Drop extra columns and rename the column to match with the API response 
                selected_columns = ['MemberID', 'LatLng', 'RouteCode', 'PickOrder', 'distance', 'DistanceToNextEmployee', 'NextLat', 'NextLon']

                in_city_emp_final_cleaned_df = RouteResponse.get_response_df(in_city_emp_final_df, selected_columns, is_manual_distance)

            if not out_city_emp_df.empty:
                out_city_emp_final_cleaned_df = process_employee_df(out_city_emp_df, center, max_emp_pickup_arc, limit_size, thr, is_manual_distance)


            in_city_emp_final_cleaned_df = in_city_emp_final_cleaned_df if 'in_city_emp_final_cleaned_df' in locals() else pd.DataFrame()

            emp_pickup_df = pd.concat([in_city_emp_final_cleaned_df, out_city_emp_final_cleaned_df], ignore_index=True) if not in_city_emp_final_cleaned_df.empty or not out_city_emp_final_cleaned_df.empty else pd.DataFrame()

            emp_pickup_df = emp_pickup_df.sort_values(by=['RouteCode', 'PickOrder'], ascending=[False, True]).reset_index(drop=True)

            final_response = RouteResponse.generate_json_response(emp_pickup_df, is_manual_distance)
            return final_response
        
        except Exception as e:
            logging.error(f'Failed to execute Master Routing: {e}')
            missing_params = {
                "CityLatLong": "empty parameter",
                "MaxEmplPickupRadius": "empty parameter",
                "FacilityLatLong": "empty parameter",
                "Members": "empty parameter"
            }

            # Check for missing parameters in the request data
            for param in missing_params:
                if param not in return_data:
                    missing_params[param] = "missing parameter"

            # Validate employee data if present
            if "Members" in return_data:
                employee_datas = pd.DataFrame(return_data['Members'])
                if employee_datas.empty:
                    missing_params['Members'] = "Given Data is Empty"

            return jsonify({"error": str(e), "missing_parameters": missing_params}), 400


class UpdateRoutePlannerAPI(Resource):

    def post(self):
        print('Loading.....')
        try:
            #Getting Json Input
            return_data = request.get_json()

            is_manual_distance = return_data['IsManualDistance']
            if isinstance(is_manual_distance, str):
                if is_manual_distance.lower() == 'true':
                    is_manual_distance = True
                elif is_manual_distance.lower() == 'false':
                    is_manual_distance = False
                else:
                    raise ValueError("IsManualDistance must be a boolean value (True or False).")
            elif not isinstance(is_manual_distance, bool):
                raise ValueError("IsManualDistance must be a boolean value.")
            
            # city Radius/boundary
            city_radius = return_data['ClusteringRadius']
            if city_radius is None or city_radius <= 0:
                raise ValueError("ClusteringRadius must be a valid positive number.")

            #Facility Lat&Lon
            latlong = return_data['FacilityLatLong']
            # try:
            center = tuple(map(float, latlong.strip().split(',')))
            latitude, longitude = center
            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude must be between -90 and 90 degrees.")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude must be between -180 and 180 degrees.")


            #Employee Details and Convert to dataframe
            emp_data = return_data['Members']
            employee = pd.DataFrame(emp_data)

            if employee['RouteCode'].isnull().all():
                raise ValueError("All values in the 'RouteCode' column are null. Use Init Route Planner API")
            
            if employee['MemberID'].isnull().any():
                count_missing = employee['MemberID'].isnull().sum()
                raise ValueError(f"{count_missing} MemberIDs have null values in 'MemberID'.")
            invalid_members = []

            for index, row in employee.iterrows():
                latlng = row['LatLng']
                member_id = row['MemberID']
                latitude, longitude = map(float, latlng.strip().split(','))
                # Check if latitude and longitude are within valid ranges
                if not (-90 <= latitude <= 90):
                    invalid_members.append(member_id)
                if not (-180 <= longitude <= 180):
                    invalid_members.append(member_id)
                if invalid_members:
                    raise ValueError(f"The following MemberIDs invalid LatLng: {', '.join(invalid_members)}.")


            if is_manual_distance:
                employee = employee.rename(columns={'ManualDistance': 'distance'})
            else:
                employee = employee.rename(columns={'DistanceToOffice': 'distance'})
            
            null_manual_distances = employee[employee['distance'].isnull()]['MemberID']
            if not null_manual_distances.empty:
                raise ValueError(f"The following MemberIDs have null values in distance: {', '.join(null_manual_distances)}.")
            
            in_city_emp_df = employee[employee['distance'] <= city_radius]
            out_city_emp_df = employee[employee['distance'] > city_radius]

           # Process grouped employees
            in_city_final_df = self.process_grouped_employees(in_city_emp_df, center, is_manual_distance)
            out_city_final_df = self.process_grouped_employees(out_city_emp_df, center, is_manual_distance, is_out_city=True)

            # Combine and sort final DataFrame
            emp_pickup_df = pd.concat([in_city_final_df, out_city_final_df], ignore_index=True)
            emp_pickup_df = emp_pickup_df.sort_values(by=['RouteCode', 'PickOrder'], ascending=[False, True]).reset_index(drop=True)

            # Generate final response
            final_response = RouteResponse.generate_json_response(emp_pickup_df, is_manual_distance)
            return final_response

        except Exception as e:
            logging.error(f'Failed to execute EmpAssignAPI: {e}')
            return self.handle_missing_parameters(return_data), 401


    def process_grouped_employees(self, emp_df, center, is_manual_distance, is_out_city=False):
        """
        Process employee DataFrame to assign route code and calculate distances.
        """
        try:
            if emp_df.empty:
                return pd.DataFrame()
                    
            # Prepare employee data for processing
            grouped_emp_df = RouteResponse.prep_new_emp_df(emp_df)

            # Assign route codes
            route_code_df = RoutePlan.assign_closest_routecode(grouped_emp_df, is_out_city)

            # Reset distances and prepare final DataFrame
            final_emp_df = RoutePlan.reset_distance_to_next_emp(route_code_df, center)

            selected_columns = ['MemberID', 'LatLng', 'RouteCode', 'PickOrder', 'distance', 'KMDistance', 'closest_stop_Latitude', 'closest_stop_Longitude']

            return RouteResponse.get_response_df(final_emp_df, selected_columns, is_manual_distance)
        except Exception as e:
            logging.error('Failed to execute function process_grouped_employees: '+ str(e))
            raise        

    def handle_missing_parameters(self, return_data):
        """
        Handle missing parameters in the request data and generate appropriate error messages.
        """
        log = {}
        missing_params = []

        # Check for all required parameters and log missing ones
        required_params = ['ClusteringRadius', 'FacilityLatLong', 'Members']
        for param in required_params:
            if param not in return_data:
                missing_params.append(param)
                log[param] = "empty parameter"

        # Additional validation for employee data
        if 'Members' in return_data and len(return_data['Members']) < 1:
            log['Members'] = "Given data is empty"

        logging.error(f'Missing parameters: {missing_params}')
        return {"error": "Missing or invalid parameters", "missing_parameters": missing_params}
    

api.add_resource(InitRoutePlannerAPI, '/route_planner', endpoint='int_route_planner')
api.add_resource(UpdateRoutePlannerAPI, '/emp_assign', endpoint='update_Route')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=PORT)

