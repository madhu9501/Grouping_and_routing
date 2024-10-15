import pandas as pd
import numpy as np

import logging
import logstash
import sys
 

import math

host = '165.22.223.79'
test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
# test_logger.addHandler(logstash.LogstashHandler( host, 8080, version=1))
test_logger.addHandler(logstash.TCPLogstashHandler( host, 8080, version=1))

class CityCircle:

    @staticmethod
    def split_circle_by_arc(radius_km, distance_between_lines_km):
        """
        Get the number of initial routes
        """
        try:
            circumference_km = 2 * math.pi * radius_km
            num_lines = circumference_km / distance_between_lines_km

            return num_lines
        except Exception as e:

            logging.error('Failed to execute function split_circle_by_arc: '+ str(e) )
            return {"message": "internal server error", "code": 501}



    @staticmethod
    def get_intersection_coordinates(center, radius_km, num_lines):
        """
        Get the co-ordinates of the start of the initial route
        """
        try:
            center_x = center[0]
            center_y= center[1]
            coordinates = []
            for i in range(int(num_lines)):
                angle_rad = 2 * math.pi * i / num_lines

                # Convert radius from kilometers to degrees
                delta_lat = radius_km / 111  # 1 degree latitude is approximately 111 km
                delta_lon = radius_km / (111 * math.cos(math.radians(center_x)))  # Adjust for latitude

                x = center_x + delta_lat * math.cos(angle_rad)
                y = center_y + delta_lon * math.sin(angle_rad)
                coordinates.append((x, y))
            return coordinates
        except Exception as e:

            logging.error('Failed to execute function get_intersection_coordinates: '+ str(e) )
            return {"message": "internal server error", "code": 501}


    @staticmethod
    def perpendicular_distance(A, B, C):
        try:
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)

            AB = B - A
            AC = C - A

            cross_product = np.cross(AB, AC)
            distance = np.linalg.norm(cross_product) / np.linalg.norm(AB)

            return distance
        except Exception as e:

            logging.error('Failed to execute function find_closest_line_to_point, perpendicular_distance: '+ str(e) )
            return {"message": "internal server error", "code": 501}

    @staticmethod
    def perpendicular_intersection(A, B, C):
        try:
                
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)

            AB = B - A
            AC = C - A

            AB_norm = np.linalg.norm(AB)
            projection = np.dot(AC, AB) / AB_norm
            projection_vector = (projection / AB_norm) * AB
            intersection = A + projection_vector

            return tuple(intersection)
        except Exception as e:

            logging.error('Failed to execute function find_closest_line_to_point, perpendicular_intersection: '+ str(e) )
            return {"message": "internal server error", "code": 501}

    @staticmethod
    def is_point_on_segment(A, B, P):
        """ Check if point P is on the line segment AB """
        try:
                
            A = np.array(A)
            B = np.array(B)
            P = np.array(P)

            on_segment = (min(A[0], B[0]) <= P[0] <= max(A[0], B[0]) and
                        min(A[1], B[1]) <= P[1] <= max(A[1], B[1]))

            return on_segment
        except Exception as e:

            logging.error('Failed to execute function find_closest_line_to_point, is_point_on_segment: '+ str(e) )
            return {"message": "internal server error", "code": 501}

    @staticmethod
    def find_closest_line_to_point(center, df, lines):

        try:
                
            A = center
            closest_lines = []
            intersection_points = []

            points = [tuple(map(float, latlng.strip().split(','))) for latlng in df['LatLng']]

            for C in points:
                closest_line = None
                min_distance = float('inf')
                closest_intersection = None

                for B in lines:
                    intersection_point = CityCircle.perpendicular_intersection(A, B, C)

                    if CityCircle.is_point_on_segment(A, B, intersection_point):
                        distance = CityCircle.perpendicular_distance(A, B, C)

                        if distance < min_distance:
                            min_distance = distance
                            closest_line = B
                            closest_intersection = intersection_point

                closest_lines.append(closest_line)
                intersection_points.append(closest_intersection)
            # # Log lengths to ensure they match
            # logging.info(f"Number of points: {len(points)}, Number of closest_lines: {len(closest_lines)}, Number of intersection_points: {len(intersection_points)}")

            # if len(closest_lines) != len(df) or len(intersection_points) != len(df):
            #     logging.error(f"Length mismatch: closest_lines: {len(closest_lines)}, intersection_points: {len(intersection_points)}, df rows: {len(df)}")
            #     return {"message": "length mismatch error", "code": 400}
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(f"closest_lines: {len(closest_lines)}")
            # print(f"closest_lines: {type(closest_lines)} \n {type(closest_lines[0])} \n {closest_lines[0]}")

            # print(f"intersection_points: {len(intersection_points)}")
            # print(f"intersection_points: {type(intersection_points)} \n {type(intersection_points[0])} \n {intersection_points[0]}")

            # print(f"df rows: {len(df)}")

            
            # df.loc[:, 'route'] = closest_lines
            # df.loc[:, 'intersection_point'] = intersection_points
            df['route'] = pd.Series(closest_lines, index=df.index)
            df['intersection_point'] = pd.Series(intersection_points, index=df.index)
            df['LatLng'] = points

            return df
        except Exception as e:

            logging.error('Failed to execute function find_closest_line_to_point: '+ str(e) )
            return {"message": "internal server error", "code": 501}

    @staticmethod
    def split_sector_to_bin(df, max_distance, bin_range, is_manual_distance):
        try:
            # Create bins of given range intervals including upper bound that exceeds the maximum DistanceToOffice
            bins = np.arange(0, max_distance + bin_range, bin_range)  # Adjusted to include max_distance
            labels = range(1, len(bins))

            # Assign group numbers based on bins
            if is_manual_distance:
                df['GroupNumber'] = pd.cut(df['ManualDistance'], bins=bins, labels=labels, right=False)
            else:
                df['GroupNumber'] = pd.cut(df['DistanceToOffice'], bins=bins, labels=labels, right=False)


            # Ensure any NaN values are handled by adding a separate category
            df['GroupNumber'] = df['GroupNumber'].cat.add_categories([labels[-1] + 1])
            df['GroupNumber'].fillna(labels[-1] + 1, inplace=True)

            return df
        except Exception as e:

            logging.error('Failed to execute function apply_split_to_routes, split_sector_to_bin: '+ str(e) )
            return {"message": "internal server error", "code": 501}
        

    @staticmethod
    def apply_split_to_routes(df, max_distance, bin_range, is_manual_distance):
        try:
            # Group by 'route' and apply the function to each group
            grouped = df.groupby('route')
            result_df = pd.DataFrame()
            
            for route, group_df in grouped:
                binned_df = CityCircle.split_sector_to_bin(group_df, max_distance, bin_range, is_manual_distance)
                result_df = pd.concat([result_df, binned_df], ignore_index=True)
            
            return result_df
        except Exception as e:

            logging.error('Failed to execute function apply_split_to_routes: '+ str(e) )
            return {"message": "internal server error", "code": 501}

    # @staticmethod
    # def split_sector_to_bin(df, max_distance, bin_range, is_manual_distance):
    #     try:
                
    #         # Create bins of given range intervals including upper bound that exceeds the maximum DistanceToOffice
    #         bins = np.arange(0, max_distance + bin_range, bin_range)  # Adjusted to include max_distance
    #         labels = range(1, len(bins))

    #         # Assign group numbers based on bins
    #         if is_manual_distance:
    #             df['GroupNumber'] = pd.cut(df['ManualDistance'], bins=bins, labels=labels, right=False)
    #         else:
    #             df['GroupNumber'] = pd.cut(df['DistanceToOffice'], bins=bins, labels=labels, right=False)
            

    #         # Ensure any NaN values are handled by adding a separate category
    #         df['GroupNumber'] = df['GroupNumber'].cat.add_categories([labels[-1] + 1])
    #         df['GroupNumber'].fillna(labels[-1] + 1, inplace=True)

    #         return df
    #     except Exception as e:

    #         logging.error('Failed to execute function split_sector_to_bin: '+ str(e) )
    #         return {"message": "internal server error", "code": 501}


        















