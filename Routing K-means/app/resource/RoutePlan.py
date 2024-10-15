import logging
import logstash
import sys
 
import uuid
import math
import pandas as pd
import numpy as np
from geopy.distance import great_circle



host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class RoutePlan:

    @staticmethod
    def calculate_distance(latlng1, latlng2):
        try:
            dist = great_circle(latlng1, latlng2).kilometers
            return dist
        except Exception as e:
            logging.error('Failed to execute function calculate_distance: '+ str(e) )
            raise



    # @staticmethod
    # def assign_pick_order(df, group_number, start_pick_order, is_manual_distance):
    #     try:
                
    #         group_df = df[df['GroupNumber'] == group_number].copy()
    #         if group_df.empty:
    #             return pd.DataFrame()

    #         # Ensure original DistanceToOffice values are used
    #         if is_manual_distance:
    #             original_distance_to_office = group_df['ManualDistance'].copy()
    #         else:
    #             original_distance_to_office = group_df['DistanceToOffice'].copy()


    #         # Initialize PickOrder
    #         group_df['PickOrder'] = np.nan
    #         pick_order = start_pick_order

    #         # Start with the farthest employee in terms of DistanceToOffice
    #         if is_manual_distance:
    #             group_df = group_df.sort_values(by='ManualDistance', ascending=False).reset_index(drop=True)
    #         else:
    #             group_df = group_df.sort_values(by='DistanceToOffice', ascending=False).reset_index(drop=True)


    #         group_df.loc[0, 'PickOrder'] = pick_order
    #         pick_order += 1

    #         while group_df['PickOrder'].isna().any():
    #             prev_employee = group_df[group_df['PickOrder'] == pick_order - 1]['LatLng'].values[0]

    #             # Calculate distance to the previous employee
    #             group_df['DistanceToPrevEmployee'] = group_df['LatLng'].apply(lambda x: RoutePlan.calculate_distance(x, prev_employee))
                
    #             # Find the next employee with the minimum distance to the previous employee
    #             next_employee_df = group_df[group_df['PickOrder'].isna()]
    #             if next_employee_df.empty:
    #                 break
                
    #             next_employee_idx = next_employee_df['DistanceToPrevEmployee'].idxmin()
                
    #             # Assign the next PickOrder to the found employee
    #             group_df.loc[next_employee_idx, 'PickOrder'] = pick_order
    #             pick_order += 1

    #         # Ensure original DistanceToOffice values are retained
    #         if is_manual_distance:
    #             group_df['ManualDistance'] = original_distance_to_office.values
    #         else:
    #             group_df['DistanceToOffice'] = original_distance_to_office.values

    #         # Drop temporary columns if they exist
    #         if 'DistanceToPrevEmployee' in group_df.columns:
    #             group_df.drop(columns=['DistanceToPrevEmployee'], inplace=True)

    #         return group_df
    #     except Exception as e:

    #         logging.error('Failed to execute function set_pick_order, assign_pick_order: '+ str(e) )
    #         raise


    # @staticmethod
    # def set_pick_order(df, is_manual_distance):
    #     try:
                
    #         result_df = pd.DataFrame()
    #         unique_routes = df['route'].unique()

    #         for route in unique_routes:
    #             route_df = df[df['route'] == route].copy()
    #             current_pick_order = 1

    #             while not route_df.empty:
    #                 farthest_group_number = route_df['GroupNumber'].max()
    #                 ordered_groups_df = RoutePlan.assign_pick_order(route_df, farthest_group_number, current_pick_order, is_manual_distance)
    #                 current_pick_order = ordered_groups_df['PickOrder'].max() + 1
    #                 result_df = pd.concat([result_df, ordered_groups_df], ignore_index=True)
    #                 route_df = route_df[route_df['GroupNumber'] != farthest_group_number]

    #         return result_df.reset_index(drop=True)
    #     except Exception as e:

    #         logging.error('Failed to execute function set_pick_order: '+ str(e) )
    #         raise

    @staticmethod
    def assign_pick_order(df, group_number, start_pick_order, is_manual_distance):
        try:
            group_df = df[df['GroupNumber'] == group_number].copy()
            if group_df.empty:
                return pd.DataFrame()

            # # Ensure original DistanceToOffice values are used
            # if is_manual_distance:
            #     original_distance_to_office = group_df['ManualDistance'].copy()
            # else:
            #     original_distance_to_office = group_df['DistanceToOffice'].copy()

            # Initialize PickOrder
            group_df['PickOrder'] = np.nan
            pick_order = start_pick_order

            # Start by sorting employees based on DistanceToOffice
            if is_manual_distance:
                group_df = group_df.sort_values(by='ManualDistance', ascending=False).reset_index(drop=True)
            else:
                group_df = group_df.sort_values(by='DistanceToOffice', ascending=False).reset_index(drop=True)
            group_df['PickOrder'] = range(pick_order, pick_order + len(group_df))

            # Update the pick order increment
            pick_order += len(group_df)

            # # Ensure original DistanceToOffice values are retained
            # if is_manual_distance:
            #     group_df['ManualDistance'] = original_distance_to_office.values
            # else:
            #     group_df['DistanceToOffice'] = original_distance_to_office.values

            return group_df
        except Exception as e:

            logging.error('Failed to execute function set_pick_order, assign_pick_order: '+ str(e) )
            raise

    def set_pick_order(df, is_manual_distance):
        try:
                
            result_df = pd.DataFrame()
            unique_routes = df['route'].unique()

            for route in unique_routes:
                route_df = df[df['route'] == route].copy()
                current_pick_order = 1

                while not route_df.empty:
                    farthest_group_number = route_df['GroupNumber'].max()
                    ordered_groups_df = RoutePlan.assign_pick_order(route_df, farthest_group_number, current_pick_order, is_manual_distance)
                    current_pick_order = ordered_groups_df['PickOrder'].max() + 1
                    result_df = pd.concat([result_df, ordered_groups_df])
                    route_df = route_df[route_df['GroupNumber'] != farthest_group_number]

            return result_df.reset_index(drop=True)
        except Exception as e:

            logging.error('Failed to execute function set_pick_order: '+ str(e) )
            raise
        

    @staticmethod
    def get_distance_to_next_emp(df, center):
        """
        Add columns for distance to next employee and distance to center.
        
        Parameters:
        - df: DataFrame containing the employee data with 'LatLng', 'PickOrder', and 'route' columns.
        - center: Tuple (latitude, longitude) for the center point.
        """
        try:
                
            # Sort DataFrame by 'route', 'GroupNumber', and 'PickOrder'
            df = df.sort_values(by=['route', 'GroupNumber', 'PickOrder'], ascending=[False, False, True]).reset_index(drop=True)
            pd.set_option('display.precision', 15)

            def get_distance_to_next_point(row, df, center):
                try:
                        
                    next_row = df[(df['route'] == row['route']) & (df['PickOrder'] == row['PickOrder'] + 1)]
                    if not next_row.empty:
                        next_lat, next_lon = next_row.iloc[0]['LatLng']
                        next_distance = RoutePlan.calculate_distance(row['LatLng'], next_row.iloc[0]['LatLng'])
                        return round(next_distance, 4), next_lat, next_lon
                    else:
                        center_lat, center_lon = center
                        center_distance = RoutePlan.calculate_distance(row['LatLng'], center)
                        return round(center_distance, 2), center_lat, center_lon
                except Exception as e:

                    logging.error('Failed to execute function get_distance_to_next_emp, get_distance_to_next_point: '+ str(e) )
                    raise
            
            df[['DistanceToNextEmployee', 'NextLat', 'NextLon']] = df.apply(
                lambda row: pd.Series(get_distance_to_next_point(row, df, center)), axis=1
            )
            # pd.reset_option('display.precision')
            return df
        except Exception as e:

            logging.error('Failed to execute function get_distance_to_next_emp: '+ str(e) )
            raise



    @staticmethod
    def generate_unique_route_codes(df, col):
        """
        Generate a unique code for each unique route and assign it to the 'RouteCode' column.
        
        Parameters:
        - df: DataFrame containing the employee data with a 'route' column.
        
        Returns:
        - DataFrame with a new 'RouteCode' column containing unique codes for each route.
        """
        try:
                
            # Create a mapping of unique routes to unique codes
            unique_routes = df[col].unique()
            route_to_code = {route: str(uuid.uuid4()) for route in unique_routes}
            
            # Map these codes to the 'RouteCode' column
            df['RouteCode'] = df[col].map(route_to_code)
            
            return df
        except Exception as e:

            logging.error('Failed to execute function generate_unique_route_codes: '+ str(e) )
            raise


    @staticmethod
    def assign_next_point(df, center_latlng):
        try:
            
            df['PickOrder'] = None
            df['DistanceToNextEmployee'] = None
            df['closest_stop_Latitude'] = None
            df['closest_stop_Longitude'] = None

            groups = df.groupby('groupNumber')
            
            for group_number, group in groups:
                if len(group) == 0:
                    continue
                
                if len(group) == 1:
                    single_row = group.iloc[0]
                    single_index = single_row.name
                    single_latlng = (single_row['latitude'], single_row['longitude'])

                    df.at[single_index, 'PickOrder'] = 1

                    df.at[single_index, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance(single_latlng, center_latlng)
                    df.at[single_index, 'closest_stop_Latitude'] = center_latlng[0]
                    df.at[single_index, 'closest_stop_Longitude'] = center_latlng[1]
                    continue

                # Convert the group DataFrame to a list of (index, (lat, lng))
                latlng_list = [(row.Index, (row.latitude, row.longitude)) for row in group.itertuples()]
                
                # Find the farthest latlng and assign PickOrder 1
                farthest_row = group.loc[group['distance'].idxmax()]



                farthest_index = farthest_row.name
                farthest_latlng = (farthest_row['latitude'], farthest_row['longitude'])

                df.at[farthest_index, 'PickOrder'] = 1
                
                remaining_latlng_list = [x for x in latlng_list if x[0] != farthest_index]
                current_latlng = farthest_latlng
                current_index = farthest_index
                pick_order = 2
                
                while remaining_latlng_list:
                    # Find the closest latlng to current_latlng
                    closest_index, closest_latlng = min(remaining_latlng_list, key=lambda x: RoutePlan.calculate_distance(current_latlng, x[1]))
                
                    closest_lat = float(closest_latlng[0])
                    closest_long = float(closest_latlng[1])

                    # Assign PickOrder and calculate distance
                    df.at[closest_index, 'PickOrder'] = pick_order
                    df.at[current_index, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance(current_latlng, closest_latlng)
                    df.at[current_index, 'closest_stop_Latitude'] = closest_lat
                    df.at[current_index, 'closest_stop_Longitude'] = closest_long
                    
                    # Update current_latlng and remove from remaining list
                    current_latlng = closest_latlng
                    current_index = closest_index
                    remaining_latlng_list = [x for x in remaining_latlng_list if x[0] != closest_index]
                    pick_order += 1

                # Handle the last employee
                if not df[df['groupNumber'] == group_number].empty:
                    # Get the last row in the sorted group
                    last_row = df[df['groupNumber'] == group_number].sort_values(by='PickOrder').iloc[-1]
                    last_index = last_row.name
                    last_latlng = (last_row['latitude'], last_row['longitude'])
                    
                    df.at[last_index, 'DistanceToNextEmployee'] = RoutePlan.calculate_distance(last_latlng, center_latlng)
                    df.at[last_index, 'closest_stop_Latitude'] = center_latlng[0]
                    df.at[last_index, 'closest_stop_Longitude'] = center_latlng[1]

            df = df.sort_values(by=['groupNumber', 'PickOrder'], ascending=[False, True]).reset_index(drop=True)
            
            return df
        except Exception as e:

            logging.error('Failed to execute function assign_next_point: '+ str(e) )
            raise

    @staticmethod
    def assign_closest_routecode(df, out_city = False):
        try:

            group1_df = df[df['GroupNumber'] == 1]  # Group with RouteCodes
            group2_df = df[df['GroupNumber'] == 2]  # Group without RouteCodes
            group1_df = group1_df.reset_index(drop=True)
            group2_df = group2_df.reset_index(drop=True)
            

            # Convert LatLng strings to tuples for both groups
            group1_df['LatLng_tuple'] = group1_df['LatLng'].apply(lambda x: tuple(map(float, x.strip().split(','))))
            group2_df['LatLng_tuple'] = group2_df['LatLng'].apply(lambda x: tuple(map(float, x.strip().split(','))))
            
            # Sort group1_df by RouteCode and PickOrder
            group1_df = group1_df.sort_values(by=['RouteCode', 'PickOrder'])

            # Iterate over each LatLng in group 2
            for idx2, row2 in group2_df.iterrows():
                min_distance = float('inf')
                closest_routecode = None
                closest_latlng = None
                closest_PickOrder = None
                # Find the closest LatLng in group 1
                for idx1, row1 in group1_df.iterrows():
                    distance = RoutePlan.calculate_distance(row1['LatLng_tuple'], row2['LatLng_tuple'])
                    
                    
                    # If a closer LatLng is found, update the closest RouteCode
                    if distance < min_distance:
                        min_distance = distance
                        closest_routecode = row1['RouteCode']
                        closest_latlng = row1['LatLng_tuple']
                        closest_PickOrder = row1['PickOrder']
                    if out_city:
                        if min_distance > 10:

                            new_route_code = str(uuid.uuid4())
                            new_pickorder = 1
                            group2_df.at[idx2, 'RouteCode'] = new_route_code
                            group2_df.at[idx2, 'PickOrder'] = new_pickorder

                            # Add row2 to group1_df
                            row2_to_add = row2.copy()
                            row2_to_add['RouteCode'] = new_route_code
                            row2_to_add['PickOrder'] = new_pickorder
                            group1_df.loc[len(group1_df)] = row2_to_add


                            # group1_df = group1_df.reset_index(drop=True)

                            continue

                # Get the subset of group1_df with the same RouteCode as the closest one
                same_routecode_df = group1_df[group1_df['RouteCode'] == closest_routecode]

                # Find LatLngs above and below the closest pick order
                above_closest = same_routecode_df[same_routecode_df['PickOrder'] < closest_PickOrder].tail(1)  # Just above
                below_closest = same_routecode_df[same_routecode_df['PickOrder'] > closest_PickOrder].head(1)  # Just below

                # Initialize second closest variables
                second_closest_latlng = None
                second_closest_PickOrder = None
                second_min_distance = float('inf')

                # Check distance to above and below points
                if not above_closest.empty:
                    dist_above = RoutePlan.calculate_distance(above_closest['LatLng_tuple'].values[0], row2['LatLng_tuple'])
                    if dist_above < second_min_distance:
                        second_min_distance = dist_above
                        second_closest_latlng = above_closest['LatLng_tuple'].values[0]
                        second_closest_PickOrder = above_closest['PickOrder'].values[0]

                if not below_closest.empty:
                    dist_below = RoutePlan.calculate_distance(below_closest['LatLng_tuple'].values[0], row2['LatLng_tuple'])
                    if dist_below < second_min_distance:
                        second_min_distance = dist_below
                        second_closest_latlng = below_closest['LatLng_tuple'].values[0]
                        second_closest_PickOrder = below_closest['PickOrder'].values[0]
                
                


                # Adjust the PickOrder
                if second_closest_PickOrder is not None and closest_PickOrder is not None:
                    new_pickorder = (closest_PickOrder + second_closest_PickOrder) / 2
                elif closest_PickOrder is not None:
                    new_pickorder = closest_PickOrder + 1  # If no second closest, place it just after the closest
                else: 
                    new_route_code = str(uuid.uuid4())
                    new_pickorder = 1
                    group2_df.at[idx2, 'RouteCode'] = new_route_code
                    group2_df.at[idx2, 'PickOrder'] = new_pickorder

                    # Add row2 to group1_df
                    row2_to_add = row2.copy()
                    row2_to_add['RouteCode'] = new_route_code
                    row2_to_add['PickOrder'] = new_pickorder
                    # group1_df = pd.concat([group1_df, row2_to_add.to_frame().T], ignore_index=True)
                    # group1_df = group1_df.append(row2_to_add, ignore_index=True)
                    group1_df.loc[len(group1_df)] = row2_to_add


                    group1_df = group1_df.reset_index(drop=True)

                    continue
                
                group1_df.loc[(group1_df['RouteCode'] == closest_routecode) & (group1_df['PickOrder'] >= new_pickorder), 'PickOrder'] += 1
                new_pickorder = math.ceil(new_pickorder)

                # Assign the closest RouteCode and PickOrder from group 1 to group 2
                group2_df.at[idx2, 'RouteCode'] = closest_routecode
                group2_df.at[idx2, 'PickOrder'] = new_pickorder

                # Add row2 to group1_df
                row2_to_add = row2.copy()
                row2_to_add['RouteCode'] = closest_routecode
                row2_to_add['PickOrder'] = new_pickorder
                group1_df.loc[len(group1_df)] = row2_to_add


                group1_df = group1_df.reset_index(drop=True)


            # Combine the two groups back into a single DataFrame
            combined_df = group1_df.sort_values(by=['RouteCode', 'PickOrder'], ascending=[ False, True]).reset_index(drop=True)
            return combined_df
        except Exception as e:

            logging.error('Failed to execute function assign_closest_routecode: '+ str(e) )
            raise


    @staticmethod
    def reset_distance_to_next_emp(df, center):
        """
        Add columns for distance to next employee and distance to center.
        
        Parameters:
        - df: DataFrame containing the employee data with 'LatLng', 'PickOrder', and 'route' columns.
        - center: Tuple (latitude, longitude) for the center point.
        """
        try:
            # pd.set_option('display.precision', 15)

            def reset_distance_to_next_point(row, df, center):
                try:
                        
                    next_row = df[(df['RouteCode'] == row['RouteCode']) & (df['PickOrder'] == row['PickOrder'] + 1)]
                    if not next_row.empty:
                        next_lat, next_lon = next_row.iloc[0]['LatLng_tuple']
                        next_distance = RoutePlan.calculate_distance(row['LatLng_tuple'], next_row.iloc[0]['LatLng_tuple'])
                        return round(next_distance, 4), next_lat, next_lon
                    else:
                        center_lat, center_lon = center
                        center_distance = RoutePlan.calculate_distance(row['LatLng_tuple'], center)
                        return round(center_distance, 2), center_lat, center_lon
                except Exception as e:

                    logging.error('Failed to execute function reset_distance_to_next_emp, reset_distance_to_next_point: '+ str(e) )
                    raise
                
            df['LatLng_tuple'] = df['LatLng'].apply(lambda x: tuple(map(float, x.strip().split(','))))

            df[['KMDistance', 'closest_stop_Latitude', 'closest_stop_Longitude']] = df.apply(
                lambda row: pd.Series(reset_distance_to_next_point(row, df, center)), axis=1
            )
            # pd.reset_option('display.precision')
            # df = df.drop(columns=['DistanceToNextEmployee', 'NextLat', 'NextLon'])

            return df
        except Exception as e:

            logging.error('Failed to execute function reset_distance_to_next_emp: '+ str(e) )
            raise



"""
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

Euclidean Approximation
def calculate_distance(coord1, coord2):
    # Calculate the distance between two latitude-longitude pairs.
    # Assuming coord1 and coord2 are (latitude, longitude) tuples
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111  # Convert to kilometers

Haversine Formula
def calculate_distance(coord1, coord2):
    # Radius of Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

"""


"""
optimized code
from sklearn.neighbors import BallTree
import numpy as np

def assign_closest_routecode(df, out_city=False):
    try:
        # Split into two groups
        group1_df = df[df['GroupNumber'] == 1].copy()
        group2_df = df[df['GroupNumber'] == 2].copy()

        # Convert LatLng strings to tuples for both groups
        group1_df['LatLng_tuple'] = group1_df['LatLng'].apply(lambda x: tuple(map(float, x.strip().split(','))))
        group2_df['LatLng_tuple'] = group2_df['LatLng'].apply(lambda x: tuple(map(float, x.strip().split(','))))

        # Extract lat/lon as separate columns for BallTree calculation
        group1_latlng = np.array(group1_df['LatLng_tuple'].tolist())
        group2_latlng = np.array(group2_df['LatLng_tuple'].tolist())

        # Create a BallTree for efficient nearest-neighbor search
        tree = BallTree(np.deg2rad(group1_latlng), metric='haversine')

        # Find nearest neighbors for each point in group2_df
        distances, indices = tree.query(np.deg2rad(group2_latlng), k=1)
        
        # Convert distances from radians to kilometers (Earth's radius ~6371 km)
        distances = distances * 6371

        # Assign closest route codes and update the pick orders
        for idx2, (distance, idx1) in enumerate(zip(distances, indices.flatten())):
            row2 = group2_df.iloc[idx2]
            row1 = group1_df.iloc[idx1]

            if out_city and distance > 10:  # Threshold of 10 km for out-of-city points
                new_route_code = str(uuid.uuid4())
                new_pickorder = 1
                group2_df.at[idx2, 'RouteCode'] = new_route_code
                group2_df.at[idx2, 'PickOrder'] = new_pickorder

                # Add row2 to group1_df
                row2_to_add = row2.copy()
                row2_to_add['RouteCode'] = new_route_code
                row2_to_add['PickOrder'] = new_pickorder
                group1_df = pd.concat([group1_df, row2_to_add.to_frame().T], ignore_index=True)
                continue

            closest_routecode = row1['RouteCode']
            closest_PickOrder = row1['PickOrder']

            # Adjust the PickOrder logic
            same_routecode_df = group1_df[group1_df['RouteCode'] == closest_routecode]
            above_closest = same_routecode_df[same_routecode_df['PickOrder'] < closest_PickOrder].tail(1)
            below_closest = same_routecode_df[same_routecode_df['PickOrder'] > closest_PickOrder].head(1)

            second_closest_PickOrder = None
            if not above_closest.empty:
                second_closest_PickOrder = above_closest['PickOrder'].values[0]
            elif not below_closest.empty:
                second_closest_PickOrder = below_closest['PickOrder'].values[0]

            if second_closest_PickOrder is not None:
                new_pickorder = (closest_PickOrder + second_closest_PickOrder) / 2
            else:
                new_pickorder = closest_PickOrder + 1

            # Increment PickOrders of subsequent rows for the same RouteCode
            group1_df.loc[
                (group1_df['RouteCode'] == closest_routecode) & (group1_df['PickOrder'] >= new_pickorder), 'PickOrder'
            ] += 1
            new_pickorder = math.ceil(new_pickorder)

            # Assign closest routecode and pickorder to group2
            group2_df.at[idx2, 'RouteCode'] = closest_routecode
            group2_df.at[idx2, 'PickOrder'] = new_pickorder

            # Add the updated row to group1_df
            row2_to_add = row2.copy()
            row2_to_add['RouteCode'] = closest_routecode
            row2_to_add['PickOrder'] = new_pickorder
            group1_df = pd.concat([group1_df, row2_to_add.to_frame().T], ignore_index=True)

        # Combine both groups and return
        combined_df = group1_df.sort_values(by=['RouteCode', 'PickOrder'], ascending=[False, True]).reset_index(drop=True)
        return combined_df

    except Exception as e:
        logging.error(f'Failed to execute function assign_closest_routecode: {str(e)}')
        raise

"""
