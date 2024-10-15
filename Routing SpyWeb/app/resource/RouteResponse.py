import pandas as pd
import numpy as np



import json

import logging
import logstash
import sys
from flask import jsonify

# from SplitLocation import SplitEmployeeLocation

host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class RouteResponse:
    
    @staticmethod
    def split_lat_lon(employee_set):
        try:
            elat = []
            elon = []
            # For each row in a varible,
            for row in employee_set['LatLng']:
                try:
                    elat.append(row.split(',')[0])
                    elon.append(row.split(',')[1])
                except:
                    elat.append(np.NaN)
                    elon.append(np.NaN)
            return elat,elon
        except:
            raise
        
    @staticmethod
    def ensure_column_types(df):
        try:
            expected_types = {
                'MemberID': str,
                'LatLng': str,
                'RouteCode': str,
                'PickOrder': int,
                'distance': float,
                'KMDistance': float,
                'closest_stop_Latitude': float,
                'closest_stop_Longitude': float
            }

            # Convert each column to the correct type
            for column, dtype in expected_types.items():
                try:
                    df[column] = df[column].astype(dtype)
                except KeyError:
                    print(f"Column '{column}' not found in DataFrame.")
                except ValueError:
                    print(f"Error converting column '{column}' to {dtype}.")
            
            return df
        except Exception as e:

            logging.error('Failed to execute function get_response_df, ensure_column_types: '+ str(e) )
            raise


    @staticmethod
    def get_response_df(df, selected_columns, is_manual_distance):
        try: 
            def format_latlng(latlng):
                """Format LatLng as a string with 15 decimal places."""
                if isinstance(latlng, tuple) and len(latlng) == 2:
                    return f"{latlng[0]:.15f}, {latlng[1]:.15f}"
                return latlng

            # Select columns
            result_df = df[selected_columns]
            


            # Rename columns as required
            result_df = result_df.rename(columns={'DistanceToNextEmployee': 'KMDistance', 'NextLat': 'closest_stop_Latitude', 'NextLon': 'closest_stop_Longitude'})
            
            # Apply formatting functions
            if 'LatLng' in result_df.columns:
                result_df['LatLng'] = result_df['LatLng'].apply(format_latlng)
            


            result_df = RouteResponse.ensure_column_types(result_df)
            if is_manual_distance:
                result_df.rename(columns={'distance': 'ManualDistance'})
            else:
                result_df.rename(columns={'distance': 'DistanceToOffice'})
                                 
            return result_df
        except Exception as e:

            logging.error('Failed to execute function get_response_df, ensure_column_types: '+ str(e) )
            raise
        

        
    @staticmethod
    def generate_json_response(df: pd.DataFrame, is_manual_distance: bool) -> str:
        """s
        Generate a JSON response from a DataFrame grouped by 'RouteCode'.
        
        Args:
        - df (pd.DataFrame): The DataFrame to process. Must contain the columns:
        'MemberID', 'LatLng', 'PickOrder', 'DistanceToOffice', 'KMDistance',
        'RouteCode', 'closest_stop_Latitude', 'closest_stop_Longitude'.
        
        Returns:
        - Response: JSON response object.
        """
        try:
            response = {"RESPONSE FORMAT": [], "result": "Success"}
            
            # Initialize a counter for route names
            route_counter = 1
            
            # Group the DataFrame by 'RouteCode'
            for route_code, group in df.groupby('RouteCode'):
                route_details = {
                    "RouteCode": route_code,
                    "RouteTotalDistanceCovered": group['KMDistance'].sum(),
                    "MembersRouteDetails": []
                }
                        # "LatLng": f"{row['LatLng'][0]}, {row['LatLng'][1]}",  # Format LatLng as a string
                
                for _, row in group.iterrows():
                    if is_manual_distance:
                        member_details = {
                            "MemberID": row['MemberID'],
                            "LatLng": row['LatLng'],  # Format LatLng as a string
                            "PickOrder": row['PickOrder'],
                            "ManualDistance": row['ManualDistance'],
                            "KMDistance": row['KMDistance'],
                            "RouteCode": row['RouteCode'],
                            "closest_stop_Latitude": row['closest_stop_Latitude'],
                            "closest_stop_Longitude": row['closest_stop_Longitude']
                        }
                    else:
                        member_details = {
                            "MemberID": row['MemberID'],
                            "LatLng": row['LatLng'],  # Format LatLng as a string
                            "PickOrder": row['PickOrder'],
                            "DistanceToOffice": row['DistanceToOffice'],
                            "KMDistance": row['KMDistance'],
                            "RouteCode": row['RouteCode'],
                            "closest_stop_Latitude": row['closest_stop_Latitude'],
                            "closest_stop_Longitude": row['closest_stop_Longitude']
                        }
                    route_details["MembersRouteDetails"].append(member_details)
                
                # Assign a sequential route name
                route_details["Name"] = f"Route-{route_counter}"
                route_counter += 1

                response["RESPONSE FORMAT"].append(route_details)
            
            # Convert to JSON response object using jsonify
            return jsonify(response)
        except Exception as e:
            logging.error('Failed to execute function generate_json_response: '+ str(e))
            raise
        
    @staticmethod
    def prep_new_emp_df(df):
        try:
            print(df.sample(10))
            df['GroupNumber'] = df['RouteCode'].apply(lambda x: 2 if pd.isnull(x) else 1)
            lat, lon = RouteResponse.split_lat_lon(df)
            df['latitude'] = [float(l) for l in lat]
            df['longitude'] = [float(l) for l in lon]
            df = df.drop(columns=['ManualDistance'])

            return df
        except Exception as e:
            logging.error('Failed to execute function prep_new_emp_df: '+ str(e))
            raise