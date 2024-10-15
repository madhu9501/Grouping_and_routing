class SplitEmployeeLocation:
    ''' Creating two different points from Employee Loocation (Latitude and Longitude) '''

    @classmethod
    def split_lat_lon(cls, employee_set):
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
            return {"message": "internal server error", "code": 501}, 501
