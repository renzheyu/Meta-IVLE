import sys
import csv
import os
from datetime import datetime
from calendar import month_name, month_abbr

class CourseData():

    def __init__(self, data_dir, filename, start_date):
        if data_dir is None:
            self.data_dir = "./"
        else:
            self.data_dir = data_dir

        self.filename = filename
    def _convert_date(self, date):
        date = date.replace(",", "")
        date = date.replace("at", "")
        date = self._standardize_date(date)
        
        return datetime.strptime(date, '%B %d %Y %I:%M%p')

    def _standardize_date(self, date):
        if date[-3] == " ":
            date = date[0:-3] + date[-2:]
        if ":" not in date:
            date = date[0:-4] + date[-3] + ":00" + date[-2:]

        month_numbers = {name: num for num, name in enumerate(month_abbr) if num}
        elements = date.split()
        if elements[0] in month_abbr:
            elements[0] = month_name[month_numbers[elements[0]]]
            date = " ".join(elements)
        
        return date
    
    def get_data(self):
        print('Loading course csv files')
        csv_file = os.path.join(self.data_dir, self.filename)
        csv_reader = csv.reader(open(csv_file, 'r', errors='ignore'))
        next(csv_reader, None)
        data = {}
        for line in csv_reader:
            attribute = line[0]
            id = line[1]

            if line[2] == "":
                week = None
            else:
                week = int(line[2])

            if line[3] == "":
                available = None
            else:
                available = self._convert_date(line[3])

            if line[4] == "":
                deadline = None
            else:
                deadline = self._convert_date(line[4])

            if line[5] == "":
                duration = None
            else:
                duration = line[5]

            data[id] = {'attribute': attribute, 'week': week, 'available': available, 'deadline': deadline,
                        'duration': duration}

        return data
    


