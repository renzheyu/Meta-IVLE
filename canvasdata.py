#__author__ = 'Jihyun Park'

import sys
import csv
import pickle
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import seaborn as sns
import pandas as pd
import os
import math
from sklearn import preprocessing

from utils import *
import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats

from coursedata import CourseData
matplotlib.style.use('ggplot')


class CanvasData():

    def __init__(self, data_dir, days_limit=35, first_day=datetime(2016,6,26,0,0,0), depth=2,
                 course_id=2230, sessionize=True):
        """

        Parameters
        ----------
        data_dir : str
            Directory where all the deidentified csv files are located in.
        days_limit : int
            The maximum number of days since the 'first_day' that we want to get.
        first_day : datetime
            First date and time of the class.
        """

        if data_dir is None:
            self.data_dir = "./"
        else:
            self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        if days_limit > 140:
            days_limit = 140
        elif days_limit < 10:
            raise('ERROR! days_limit should be larger than 10. (in get_avg_clicks_per_day_category())')
        self.days_limit = days_limit

        if first_day is None:
            raise("ERROR! first_day can't be None type!")
        self.first_day = first_day

        self.depth = depth
        self.course_id = course_id

        self.csvdata = None
        self.n_students = None
        self.id2idx = None
        self.idx2id = None

        self.assignment_deadlines = None

        self.demo_info = {}
        self.demo_info_df = None # demographic info in pandas dataframe
        #                         (only using it for the 2230 class at the moment, 8/2/2017)

        # Load csv files
        self._load_csv()
        self._load_duedates_and_points()
        #self._load_files_info()
        self._load_lecture_dates()
        self._load_exam_dates()
        self._load_demographic_info()
        '''
        if len(self.demo_info) > 0:
            self._load_grade_info()
        else:
            self.idx2grade = None'''
        self._load_grade_info()

 #       course = CourseData(self.data_dir, "course.csv", first_day)
 #       self.course_data = course.get_data()


    def _load_csv(self):
        """
        The de-identified CSV files should have the following columns
            random_id, url_action, created_at, interaction_seconds, ip_address
            
        Within each CSV file, the rows should come in reverse chronological order

        """

        print('Loading csv files')
        deidentified_data_dir = os.path.join(self.data_dir, 'deidentified')
        data = {}
        index = 0
        idx2id = []
        id2idx = {}
        ip2id = {}
        date_format = "%Y-%m-%dT%H:%M:%SZ"

        for file in os.listdir(deidentified_data_dir):
            url = []
            category = []
            action = []
            created_at = []
            interact_secs = []
            remote_ip = []

            if file.endswith('csv'):
                csv_file = os.path.join(deidentified_data_dir, file)
                csv_reader = csv.reader(open(csv_file, 'r'))
                next(csv_reader, None)
                for line in csv_reader:
                    if len(line) != 0:
                        id = line[0]
                        ip = line[-1]
                        url.append(line[1])
                        category.append(self.get_cats_from_url(line[1], self.depth))
                        action.append(line[2])
                        # you need to subtract 7 hours to get the california time
                        created_at.append(datetime.strptime(line[3], date_format) - timedelta(0, 25200))
                        interact_secs.append(line[4])
                        remote_ip.append(ip)
                        if ip2id.get(ip, None) is None:
                            ip2id[ip] = []
                        if int(id) not in ip2id[ip]:
                            ip2id[ip].append(int(id))

                id = int(id)
                data[id] = {"url": url[::-1], "category": category[::-1], "action": action[::-1],
                            "created_at": created_at[::-1],
                            "interact_secs": interact_secs[::-1], "remote_ip": remote_ip[::-1]}
                id2idx[id] = index
                idx2id.append(id)
                index += 1

        self.csvdata = data
        self.n_students = index
        self.id2idx = id2idx
        self.idx2id = idx2id
        self.ip2id = ip2id
        print('Finished Loading')


    def _load_duedates_and_points(self):
        print('Loading deadlines')

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        as_deadlines_pkl_file = os.path.join(self.data_dir, 'assignments_duedates_and_points.pkl')
        qz_deadlines_pkl_file = os.path.join(self.data_dir, 'quizzes_duedates_and_points.pkl')
        self.assignment_deadlines = {}
        self.quiz_deadlines = {}

        if os.path.exists(as_deadlines_pkl_file):
            assgnmnt_due_dates = pickle.load(open(as_deadlines_pkl_file, 'rb'))
            ## n_dt_pt : assignment_name, due_date, points
            self.assignment_deadlines_and_pnts = {n_dt_pt[0]:[datetime.strptime(n_dt_pt[1], date_format), n_dt_pt[2]]
                                                  for id, n_dt_pt in assgnmnt_due_dates.items() if n_dt_pt[1] is not None}
            self.assignment_deadlines= {n_dt_pt[0]:datetime.strptime(n_dt_pt[1], date_format)
                                        for id, n_dt_pt in assgnmnt_due_dates.items() if n_dt_pt[1] is not None}

        if os.path.exists(qz_deadlines_pkl_file):
            quiz_due_dates = pickle.load(open(qz_deadlines_pkl_file, 'rb'))
            self.quiz_deadlines_and_pnts = {n:[datetime.strptime(dt_pt[0], date_format), dt_pt[1]]
                                            for n, dt_pt in quiz_due_dates.items() if dt_pt[0] is not None}

            self.quiz_deadlines= {n:datetime.strptime(dt_pt[0], date_format) for n, dt_pt in quiz_due_dates.items() if dt_pt[0] is not None}
        # self.deadline_days = [(dt - self.first_day).days for dt in self.assignment_deadlines.values()]

        # Combine these two.
        # If there are deadlines with the same title, they will be merged. (It is also possible that they are the same.)
        if sys.version_info >= (3, 0):
            self.assignment_quiz_deadlines = dict(self.assignment_deadlines.items() | self.quiz_deadlines.items())
        else:
            self.assignment_quiz_deadlines = dict(self.assignment_deadlines.items() + self.quiz_deadlines.items())



    def _load_lecture_dates(self, lecture_days = (0, 2, 4)):
        """

        Parameters
        ----------
        lecture_days : Days of the week when the lectures were held. 0 is Monday and 6 is Sunday.

        Returns
        -------

        """
        print('Loading lecture dates')
        # lecture_dates = []
        # US_holidays = holidays.US()
        # for d in range(85):  # 9-10 weeks of lectures + 5 days (in case we have wrong 'first_day'.)
        #     date = self.first_day + timedelta(d)
        #     wday = date.weekday() # Day of the week. 0 is Monday
        #     if wday in lecture_days:
        #         if date not in US_holidays:
        #             lecture_dates.append(date)
        if self.course_id == 1112:
            self.lecture_dates = [datetime(2016, 1, 4, 11),  datetime(2016, 1, 6, 11),
                                  datetime(2016, 1, 8, 11),  datetime(2016, 1, 11, 11),
                                  datetime(2016, 1, 13, 11), datetime(2016, 1, 15, 11),
                                  datetime(2016, 1, 20, 11), datetime(2016, 1, 22, 11),
                                  datetime(2016, 1, 27, 11), datetime(2016, 1, 29, 11),
                                  datetime(2016, 2, 1, 11),  datetime(2016, 2, 3, 11),
                                  datetime(2016, 2, 5, 11),  datetime(2016, 2, 8, 11),
                                  datetime(2016, 2, 10, 11), datetime(2016, 2, 12, 11),
                                  datetime(2016, 2, 17, 11), datetime(2016, 2, 22, 11),
                                  datetime(2016, 2, 24, 11), datetime(2016, 2, 26, 11),
                                  datetime(2016, 2, 29, 11), datetime(2016, 3, 2, 11),
                                  datetime(2016, 3, 4, 11),  datetime(2016, 3, 9, 11),
                                  datetime(2016, 3, 11, 11)]

            self.first_lec_date_in_week = [datetime(2016, 1, 4, 11), datetime(2016, 1, 11, 11),
                                           datetime(2016, 1, 20, 11), datetime(2016, 1, 27, 11),
                                           datetime(2016, 2, 1, 11), datetime(2016, 2, 8, 11),
                                           datetime(2016, 2, 17, 11), datetime(2016, 2, 22, 11),
                                           datetime(2016, 2, 29, 11), datetime(2016, 3, 9, 11)]
            self.lecture_minutes = 50  # lectures 50 minutes. This is hard coded at this point

        elif self.course_id == 2230:
            self.lecture_dates = [datetime(2016,6,20,7,0), datetime(2016,6,21,7,0),
                                  datetime(2016,6,22,7,0), datetime(2016,6,23,7,0),
                                  datetime(2016,6,24,7,0), datetime(2016,6,27,7,0),
                                  datetime(2016,6,28,7,0), datetime(2016,6,29,7,0),
                                  datetime(2016,6,30,7,0), datetime(2016,7,1, 7,0),
                                  datetime(2016,7, 4,7,0), datetime(2016,7,5, 7,0),
                                  datetime(2016,7, 6,7,0), datetime(2016,7,7, 7,0),
                                  datetime(2016,7, 8,7,0), datetime(2016,7,11,7,0),
                                  datetime(2016,7,12,7,0), datetime(2016,7,13,7,0),
                                  datetime(2016,7,14,7,0), datetime(2016,7,15,7,0),
                                  datetime(2016,7,18,7,0), datetime(2016,7,19,7,0),
                                  datetime(2016,7,20,7,0), datetime(2016,7,21,7,0),
                                  datetime(2016,7,22,7,0), datetime(2016,7,23,7,0)]
                                  # last entry padded to calculate pre/post

            # last entry added for deadline for last week (for function get_file_due_date_from_url() in prepostPhysics)
            self.first_lec_date_in_week = [self.lecture_dates[0], self.lecture_dates[5],
                                           self.lecture_dates[10], self.lecture_dates[15],
                                           self.lecture_dates[20], datetime(2016,7,25,7,0)]

            self.challenges_due_dates = [datetime(2016,6,22,23,59,59), datetime(2016,6,26,23,59,59),
                                         datetime(2016,6,29,23,59,59), datetime(2016,7, 3,23,59,59),
                                         datetime(2016,7, 6,23,59,59), datetime(2016,7,10,23,59,59),
                                         datetime(2016,7,13,23,59,59), datetime(2016,7,17,23,59,59),
                                         datetime(2016,7,20,23,59,59), datetime(2016,7,24,23,59,59)]
            # From 7AM to Midnight (Daily homework is due at midnight.
            # Students are allowed to take the lecture anytime before midnight)
            self.lecture_minutes = 1020
            student_lec_dates = []

        elif self.course_id == 5548:
            self.lecture_dates = [datetime(2017, 6, 26, 0, 0), datetime(2017, 6, 27, 0, 0),
                                  datetime(2017, 6, 28, 0, 0), datetime(2017, 6, 29, 0, 0),
                                  datetime(2017, 6, 30, 0, 0), datetime(2017, 7,  3, 0, 0),
                                  datetime(2017, 7,  4, 0, 0), datetime(2017, 7,  5, 0, 0),
                                  datetime(2017, 7,  6, 0, 0), datetime(2017, 7,  7, 0, 0),
                                  datetime(2017, 7, 10, 0, 0), datetime(2017, 7, 11, 0, 0),
                                  datetime(2017, 7, 12, 0, 0), datetime(2017, 7, 13, 0, 0),
                                  datetime(2017, 7, 14, 0, 0), datetime(2017, 7, 17, 0, 0),
                                  datetime(2017, 7, 18, 0, 0), datetime(2017, 7, 19, 0, 0),
                                  datetime(2017, 7, 20, 0, 0), datetime(2017, 7, 21, 0, 0),
                                  datetime(2017, 7, 24, 0, 0), datetime(2017, 7, 25, 0, 0),
                                  datetime(2017, 7, 26, 0, 0), datetime(2017, 7, 27, 0, 0),
                                  datetime(2017, 7, 28, 0, 0), datetime(2017, 7, 29, 0, 0)]
                                  # last entry padded to calculate pre/post
            self.first_lec_date_in_week = [self.lecture_dates[0], self.lecture_dates[5],
                                           self.lecture_dates[10], self.lecture_dates[15],
                                           self.lecture_dates[20], datetime(2017,7,31,0,0)]

    def _load_exam_dates(self):
        """
        Hard coded exam dates.
        Returns
        -------

        """
        if self.course_id == 1112:
            midterm1 = datetime(2016, 1, 25, 11, 0, 0)
            midterm2 = datetime(2016, 2, 19, 11, 0, 0)
            midterm3 = datetime(2016, 3, 7, 11, 0, 0)
            final = datetime(2016, 3, 18, 8, 0, 0)
            self.exam_dates = [midterm1, midterm2, midterm3, final]
            self.final_minutes = 120  # 2 hours of final exam

        elif self.course_id == 2230:
            # midterm1 = datetime()
            final = datetime(2016, 7, 26, 19, 0)
            self.exam_dates = [final]
            self.final_minutes = 120  # 2 hours of final exam


    def _load_files_info(self):
        print('Loading files information')
        pkl_file = os.path.join(self.data_dir, 'files.pkl')

        if not os.path.exists(pkl_file):
            print(pkl_file + ' does not exist!')
            return

        with open(pkl_file, 'rb') as f:
            files = pickle.load(f)

        self.file2fid = {file['display_name'].lower(): file['id'] for file in files}
        self.fid2file = {file['id']: file['display_name'].lower() for file in files}
        self.lec_files = []
        self.exam_files = []
        self.other_files = []
        self.reading_files = []
        self.hw_quiz_files = []

        if self.course_id == 1112:
            for fname in self.file2fid.keys():
                if 'lecture' in fname:
                    self.lec_files.append(fname)
                elif 'week' in fname:
                    self.reading_files.append(fname)
                elif ('mid-term' in fname) or ('final' in fname):
                    self.exam_files.append(fname)
                else:
                    self.other_files.append(fname)

        elif self.course_id == 2230:
            for fname in self.file2fid.keys():
                if 'lecture' in fname:
                    self.lec_files.append(fname)
                elif 'challenge' in fname:
                    self.hw_quiz_files.append(fname)
                elif 'quiz' in fname:
                    self.hw_quiz_files.append(fname)
                elif ('midterm' in fname) or ('final' in fname):
                    self.exam_files.append(fname)
                elif 'mf' in fname: # MF : Monday to Friday, Mastering Physics
                    self.hw_quiz_files.append(fname)
                elif 'sols' in fname:
                    self.hw_quiz_files.append(fname)
                else:
                    self.other_files.append(fname)

        self.lec_files.sort(key=natural_keys)
        self.reading_files.sort(key=natural_keys)
        self.exam_files.sort()
        self.hw_quiz_files.sort()
        self.other_files.sort()
        self.fidx2file = self.lec_files + self.reading_files + self.exam_files + \
                         self.hw_quiz_files + self.other_files
        self.file2fidx = {fname:fidx for fidx, fname in enumerate(self.fidx2file)}


    def _load_demographic_info(self):
        print('Loading demographic information including grades')
        demo_data = {}
        if self.course_id == 1112:
            csv_file = os.path.join(self.data_dir, 'all_students.csv')
        else: #self.course_id == 2230 or self.course_id == 5548:
            csv_file = os.path.join(self.data_dir, 'deidentified_gradebook.csv')
            ### THIS IS TEMPORARY
            self.demo_info_df = pd.read_csv(os.path.join(self.data_dir, 'demographic_info.csv'), engine='python')
        #else:
 #           csv_file = os.path.join(self.data_dir, 'demographic_info.csv')

        if os.path.exists(csv_file):
            csv_reader = csv.reader(open(csv_file, 'r'))
            if sys.version_info >= (3, 0):
                header = csv_reader.__next__()
            else:
                header = csv_reader.next()
            header = header[1:]  # Don't include the student id
            for line in csv_reader:
                id = int(line[0])
                d = {}
                for i, item in enumerate(line[1:]):
                    if item == '--':
                        d[header[i]] = ''
                    else:
                        d[header[i]] = item
                demo_data[id] = d

        # It will remain empty if the demographic csv data does not exist.
        self.demo_info = demo_data


    def _load_grade_info(self):
        print('Loading students final grade information (grade2id, id2grade, idx2grade, and gradebook)')
        self.g2pnts = {'A+':4.3, 'A':4.0, 'A-':3.7, 'B+':3.3, 'B':3.0, 'B-':2.7,
                           'C+':2.3, 'C':2.0, 'C-':1.7, 'D+':1.3, 'D':1.0, 'D-':0.7, 'F':0, '':''}
        # self.grades_simplified = ['F', 'D', 'C', 'B', 'A']
        self.g2gsimplepnts = {'A+': 4, 'A': 4, 'A-': 4, 'B+': 3, 'B': 3, 'B-': 3,
                                'C+': 2, 'C': 2, 'C-': 2, 'D+': 1, 'D': 1, 'F': 0, '': -1, 'P': -1}
        self.grade2id = defaultdict(list)
        self.id2grade = {}
        self.idx2grade = []
        self.idx2gradepnts = []
        '''
        for idx, sid in enumerate(self.idx2id):
            if sid not in self.demo_info.keys():
                gr = ''
            else:
                demo = self.demo_info[sid]
                gr = demo['grade']
            self.idx2grade.append(gr)
            self.idx2gradepnts.append(self.g2gsimplepnts[gr])
            if gr != '':
                self.id2grade[sid] = gr
                self.grade2id[gr].append(sid)
        
        if self.course_id == 1112:
            gradebook = {}
            csv_file = os.path.join(self.data_dir, 'gradebook.csv')
            csv_reader = csv.reader(open(csv_file, 'r'))
            header = csv_reader.next()
            possible_pts = csv_reader.next()
            header = header[1:-1]  # Don't include the student id (first column), and the Total (last column)
            for line in csv_reader:
                id = int(line[0])
                d = {h.replace(" ", "_").lower():float(line[i+1]) for i, h in enumerate(header)}
                d['final_grade'] = self.demo_info[id]['grade']
                gradebook[id] = d
            gradebook['possible_pts'] = {h.replace(" ", "_").lower():float(possible_pts[i+1]) for i, h in enumerate(header)}
            self.gradebook = gradebook
        else:
            self.gradebook = None
        '''
        ids = self.demo_info_df["roster_randomid"]
        if self.data_dir in ['./16S1 PHY 3A', './16Fa CHEM 1P', './18Wi BIO SCI 9B']:
            grades = self.demo_info_df["grade"]
        elif self.data_dir == './17S1 PHY 3A':
            grades = self.demo_info_df["post_egrade"]
        elif self.data_dir == './18S1 BIO SCI 9B':
            grades = self.demo_info_df["gr_gradef"]
        elif self.data_dir == './18S1 CHEM 1C':
            grades = self.demo_info_df["lettergrade"]
        else:
            grades = self.demo_info_df["gradebook_grade"]
            
        for i in range(len(ids)):
            if grades[i] in self.g2pnts and not pd.isnull(grades[i]):
                self.id2grade[ids[i]] = grades[i]

    def get_grade_for_id(self, student_id, number):
        grades = []
        for i in number:
            id = student_id[i]
            if id in self.id2grade:
                grades.append(self.g2pnts[self.id2grade[id]])

        return grades

    def get_num_clicks_per_day(self, student_data, type='all'):
        """
        Parameters
        ----------
        student_data : dict
            dictionary for each student. (One entry of canvas_data.) It should have 'created_at' entry.

        Returns
        -------
            np.array
            numpy array with length 'days_limit'
            Histogram (counts) of the student as a function of time.

        """
        days_limit = self.days_limit
        first_day = self.first_day
        hist_array = np.zeros(days_limit, dtype=np.int32)
        array = np.zeros(days_limit, dtype=np.int32) 
        if type == 'all':  # Default
            for time in student_data['created_at']:
                delta = time - first_day
                if delta.days < days_limit and delta.days > 0:
                    hist_array[delta.days] += 1
                    array[delta.days] += 1
        else:
            for i, time in enumerate(student_data['created_at']):
                cat = self.get_cats_from_url(student_data['url'][i], depth=1)
                if cat == type:
                    delta = time - first_day
                    if delta.days < days_limit and delta.days > 0:
                        hist_array[delta.days] += 1
        return hist_array, array

    def get_num_clicks_per_day_mat(self, type='all'):
        """
        Get (num_student X num_days) matrix
        where each row is the number of click events per day for each student.
        Returns
        -------
            np.array

        """
        clicks_per_day_mat = np.zeros((self.n_students, self.days_limit), dtype=np.int32)
        for idx, random_id in enumerate(self.idx2id):
            student = self.csvdata[random_id]
            clicks_per_day_mat[idx] = self.get_num_clicks_per_day(student, type)
        return clicks_per_day_mat

    def get_num_file_clicks_per_day(self, student_data):
        """
        Inner function for 'get_num_file_clicks_per_day_mat()'

        Parameters
        ----------
        student_data : dict

        Returns
        -------
        np.array
        row vector. the number of file clicks per day for the student.

        """
        hist_array = np.zeros(self.days_limit, dtype=np.int32)
        for time, url in zip(student_data['created_at'], student_data['url']):
            delta = time-self.first_day
            if self.is_file_url(url) and (0 <= delta.days < self.days_limit):
                hist_array[delta.days] += 1
        return hist_array

    def get_num_file_clicks_per_day_mat(self):
        """
        Get (num_students X num_days) matrix,
        where each row is the number of file click events per day for each student.

        Returns
        -------
        np.array

        """
        fclicks_per_day_mat = np.zeros((self.n_students, self.days_limit), dtype=np.int32)
        for idx, rid in enumerate(self.idx2id):
            fclicks_per_day_mat[idx] = self.get_num_file_clicks_per_day(self.csvdata[rid])

        return fclicks_per_day_mat

    def get_num_clicks_per_day_category(self, student_data, categories):
        """
        This is used inside 'get_avg_num_clicks_per_day()'

        Parameters
        ----------
        student_data : dict
            dictionary for each student. (One entry of canvas_data.) It should have 'created_at' entry.
        categories : list[string]
            list of categories that we want to get the click histogram of.

        Returns
        -------
            List[numpy array]
            List of numpy arrays with length 'days_limit'
            Each array corresponds to the histogram (counts) of the student for each category.
            The order is the same as in the argument 'categories'
        """
        days_limit = self.days_limit
        first_day = self.first_day

        # Initialize the arrays that we will return
        num_cats = len(categories)
        list_of_arrays = []
        for j in range(num_cats):
            list_of_arrays.append(np.zeros(days_limit, dtype=np.int32))
        cat2idx = {y: x for x, y in enumerate(categories)}

        for i, cat in enumerate(student_data["category"]):
            if cat is not None:
                delta = student_data["created_at"][i] - first_day
                if delta.days < days_limit:
                    if cat in categories:
                        cat_idx = cat2idx[cat]
                        list_of_arrays[cat_idx][delta.days] += 1
                    elif 'assignment' in cat or 'quiz' in cat:
                        if 'assignment' in ' '.join(categories):
                            cat_idx = cat2idx['assignment_quiz']
                            list_of_arrays[cat_idx][delta.days] += 1

        return list_of_arrays

    def get_avg_num_clicks_per_day(self, plot=True, plot_filename='./clicks_avg.pdf'):
        """
        Get the average number of clicks among the students in 'canvas_data'

        Parameters
        ----------
        plot : bool
            Plot and save the plot if True.

        Returns
        -------

        """
        days_limit = self.days_limit

        print('Getting the average number of clicks per day')
        # Get number of clicks per day for each student
        clicks_per_day_arr = self.get_num_clicks_per_day_mat()
        avg_clicks = clicks_per_day_arr.sum(axis=0) / float(self.n_students)

        if plot:
            folder, filename = os.path.split(plot_filename)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            print('Plotting..')
            plt.bar(range(days_limit), avg_clicks[:days_limit], 0.7)
            for monday in range(3,90,7):
                plt.axvline(x=monday, alpha=0.8, color='r')
            plt.xlabel('Days since Jan 1, 2016') #CHANGE TO BE GENERIC
            plt.ylabel('Average Clicks')
            plt.savefig(plot_filename)
            plt.clf()

        return clicks_per_day_arr, avg_clicks



    def get_feature_vec(self, student_data, categories):
        """
        Get a feature vector where a feature vector for a student is
        a number of clicks in 'categories', concatenated for each category.

        Currently not used. This method is not very helpful.

        Parameters
        ----------
        student_data
        categories

        Returns
        -------

        """
        list_of_arrays = self.get_num_clicks_per_day_category(student_data, categories)
        result_vec = np.array([], dtype=np.int32)
        for arr in list_of_arrays:
            result_vec = np.concatenate((result_vec, arr))
        return result_vec


    def get_cnt_feature_matrix(self, categories):
        """
        Return a matrix where each row corresponds to each student's clickstreams for 'categories'.
        The size will be (n_students, n_categories x n_days)

        Parameters
        ----------
        categories

        Returns
        -------

        """

        print('Generating feature matrix')
        days_limit = self.days_limit
        n_cats = len(categories)
        feature_mat = np.zeros((self.n_students, days_limit*n_cats), dtype=np.int32)
        for id in self.csvdata:
            idx = self.id2idx[id]
            student = self.csvdata[id]
            feature_mat[idx,:] = self.get_feature_vec(student, categories)

        return feature_mat

    def get_time_on_task_per_student(self, student, threshold, data):
        """
        Used inside get_time_on_task

        Parameters
        ----------
        student
        threshold: how many seconds should be allowed per task before cut off
        
        Returns
        -------

        """
    
        task_times = dict()
        for t in data:
            clicks = data[t]
            task_time = []
            for i in range(len(clicks)-1):
                click1 = clicks[i+1]
                click2 = clicks[i]
                difference = click1-click2
                if difference.total_seconds() > threshold:
                    difference = timedelta(seconds = threshold)
                task_time.append(difference.seconds)
            task_times[t] = sum(task_time)
        return task_times

    def get_time_on_task(self, data, threshold):   
        """
        Determines the amount of time spent on each task
        if days are not consecutive, the time on task is denoted as 'unknown'

        Parameters
        ----------
        
        Returns
        -------

        """
        sum_time = dict()
        for id in self.csvdata:
            times = data[id]
            for t in times:
                task_time = self.get_time_on_task_per_student(student, threshold, t)

            sum_time[id] = task_time.sum()
        
            

    def get_num_clicks_att(self, type_data="total"):
        """
        Returns the number of clicks per student according to a specificed type of
        data
        
        Parameters
        ----------
        type_data: "day", "category", "module", "week_module", "week_cal", "total"

        Returns
        -------
        specified data
        """
        attribute = dict()
        data = dict()
        k = 1
        for id in self.csvdata:
            student = self.csvdata[id]
            times = student["created_at"]
            if type_data == "total":
                attribute[id] = len(times)
                self.csvdata[id]["total_clicks"] = len(times)
            elif type_data == "day":
                clicks = dict()
                for time in times:
                    t = time.isoformat()[0:10]
                    if t not in clicks:
                        clicks[t] = 1
                    else:
                        clicks[t] += 1
                    
                attribute[id] = clicks
                self.csvdata[id]["clicks_per_day"] = clicks
            elif type_data == "week_module":
                week = dict()
                student = self.csvdata[id]
                if "click_week" not in student:
                    self.click_to_week()
                for click in student["click_week"]:
                    if click != None:
                        if click not in week:
                            week["week " + str(click)] = 1
                        else:
                            week["week " + str(click)] += 1
                attribute[id] = week
                self.csvdata[id]["clicks_per_week_module"] = week
            elif type_data == "week_cal":
                weeks = {"week 1": 0}
                time_spent = {"week 1": []}
                student = self.csvdata[id]
                week_num = 1
                week = self.first_day + timedelta(days=7, hours=23, minutes=59, seconds=59)
                student = self.csvdata[id]
                for time in student["created_at"]:
                    if time < week:
                        weeks["week "+ str(week_num)] += 1
                        time_spent["week "+ str(week_num)].append(time)
                    else:
                        week_num += 1
                        week = week + timedelta(days=7, hours=23, minutes=59, seconds=59)
                        weeks["week " + str(week_num)] = 1
                        time_spent["week " + str(week_num)] = []
                attribute[id] = weeks
                data[id] = time_spent
                self.csvdata[id]["clicks_per_week_cal"] = weeks
            elif type_data == "category":
                cats = dict()
                c = student["category"]
                time_spent = dict()
                for cat in range(len(c)):
                    if c[cat] not in cats and c[cat] in ['assignments', 'discussion_topics', 'files', 'pages']:
                        cats[c[cat]] = 1
                        time_spent[c[cat]] = [student["created_at"][cat]]
                    elif c[cat] in cats:
                        cats[c[cat]] += 1
                        time_spent[c[cat]].append(student["created_at"][cat])
                data[id] = time_spent
                attribute[id] = cats
                self.csvdata[id]["clicks_per_cat"] = cats
                
        return attribute, data
        
    def get_number_session(self, id, types):
        
        week_sessions = {"week 1": 0}
        start_times = []
        student = self.csvdata[id]
        times = student["created_at"]
        t = 1
        if times != None and len(times) > 0:
            start = times[0]
            start_times.append(start)
            while t <= len(times) - 1:
                diff = times[t] - times[t-1]
                if diff > timedelta(minutes = 30):
                    start = times[t]
                    start_times.append(start)
                t += 1

            if types == "total":
                return len(start_times)
            else:
                week_num = 1
                week = self.first_day + timedelta(days=7, hours=23, minutes=59, seconds=59)
                student = self.csvdata[id]
                for time in start_times:
                    if time < week:
                        week_sessions["week "+ str(week_num)] += 1
                    else:
                        week_num += 1
                        week = week + timedelta(days=7, hours=23, minutes=59, seconds=59)
                        week_sessions["week " + str(week_num)] = 1
                return week_sessions
                
        else:
            if types == "total":
                return 0
            else:
                return {"week 1": 0, "week 2": 0, "week 3": 0, "week 4": 0, "week 5": 0, "week 6": 0, "week 7": 0, "week 8": 0, "week 9": 0, "week 10": 0} 

                
                    
    def click_to_week(self):
        """
        Determines the week that is associated with each click. If click is not
        associated with a week, the value is none. 
        
        Parameters
        ----------

        Returns
        -------
        """
        
        for id in self.csvdata:
            click_week = []
            student = self.csvdata[id]
            urls = student["url"]
            for url in urls:
                if "module_item_id" in url:
                    module_id = url[-5:]
                    if module_id in self.course_data:
                        click_week.append(self.course_data[module_id]["week"])
                    else:
                        click_week.append(None)
                else:
                    click_week.append(None)

            self.csvdata[id]["click_week"] = click_week

    def time_til_deadline(self):
        """
        Determines the difference between a click and the deadline. If there is
        no deadline, the value is 0. 
        
        Parameters
        ----------

        Returns
        ---
        """
        for id in self.csvdata:
            time_diff = []
            student = self.csvdata[id]
            urls = student["url"]
            clicks = student["created_at"]
            for i in range(len(urls)):
                if "module_item_id" in urls[i]:
                    module_id = urls[i][-5:]
                    if module_id in self.course_data:
                        deadline = self.course_data[module_id]["deadline"]
                        if deadline != None:
                            diff = deadline - clicks[i]
                            time_diff.append(diff)
                        else:
                            time_diff.append(None)
                    else:
                        time_diff.append(None)
            self.csvdata[id]["time_til_deadline"] = time_diff

    def get_average_time_til_deadline(self):
        """
        Determines average time from deadline. Value is none if no time from
        deadline was calculated
        
        Parameters
        ----------

        Returns
        ----
        """
        self.time_til_deadline()
        total = timedelta(0)
        count = 0
        for id in self.csvdata: 
            time_diff = self.csvdata[id]["time_til_deadline"]
            for time in time_diff:
                if time != None:
                    total += time
                    count += 1
            self.csvdata[id]["avg_time_til_deadline"] = total/count


    def get_num_clicks_per_day_per_cat(self, student):
        cats = set(student["category"])
        counts = dict()
        for cat in cats:
            x = self.get_num_clicks_per_day(student, type=cat)
            print(x)
            
            
        
    def get_click_distribution_over_cats(self, plot=True, plot_filename='/graphs/click_distribution.pdf'):
        cat_counts = dict()
        categories = []
        for id in self.csvdata:
            student = self.csvdata[id]
            cats = set(student["category"])
            for cat in cats:
                if cat == None:
                    cat = "no_category"
                if cat not in cat_counts:
                    cat_counts[cat] = self.get_num_clicks_per_day(student, type=cat)
                else:
                    cat_counts[cat] += self.get_num_clicks_per_day(student, type=cat)

        for cat in cat_counts:
            '''
            threshold = -1#0.5*len(cat_counts[cat])
            print(cat_counts[cat], cat)
            if len(cat) <= 20 and np.where(cat_counts[cat]>100)[0].shape[0] > threshold:
                categories.append(cat)'''

            if cat in ['assignments', 'discussion_topics', 'files', 'pages']:
                categories.append(cat)
       
        if plot:
            folder, filename = os.path.split(plot_filename)
            if not os.path.isdir(folder):
                os.makedirs(folder)

            print('Plotting..')
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111)
            for cat in categories:
                ax.plot(cat_counts[cat]/float(self.n_students), label=cat)
            '''
            colormap = plt.cm.CMRmap
            colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])'''

            ax.legend()

            plt.ylabel('Average number of clicks')
            plt.xlabel('Days')
            plt.savefig(plot_filename)
            plt.clf()
        

        
        return cat_counts, categories

    def get_clicks_per_cat_mat(self, category):
        """
        Parameters
        ----------
        category: category to record clicks
        
        Get (num_studnets X num_days) matrix
        where each row is the number of clicks of a category per day for each student 

        Returns
        -------
        np.array
        
        """
        l = []
        clicks_per_cat_mat = np.zeros((self.n_students, self.days_limit), dtype = np.int32)
        for idx, rid in enumerate(self.idx2id):
            clicks_per_cat_mat[idx] = self.get_num_clicks_per_day(self.csvdata[rid], type=category)
            l.append(rid)
        return clicks_per_cat_mat, l

    def plot_clicks_per_cat_per_stud(self, id, category, student_id, change):
        student = self.csvdata[id]
        print("Plotting....")

        fig = plt.figure(figsize=(15, 10))

        
        clicks = self.get_num_clicks_per_day(student, type= category)
        plt.bar(range(self.days_limit), clicks, 0.3, label=category, color="#66A7C5")
        changepoint = change[student_id.index(id)]
        plt.axvline(x=changepoint, alpha=0.8, color='r')
        plt.xlabel('Day')
        plt.ylabel('Number of Clicks')
        plt.title("Number of Clicks for Student #" + str(id) + " in " + category.capitalize())
        plt.savefig("./" + self.data_dir + "/graphs/" + str(id) + "_" + category + ".pdf")
        plt.clf()

    def save_clicks_cat_matrix(self):
        student_id = dict()
        categories = self.get_click_distribution_over_cats(plot=False)[1]
        for cat in categories:
            mat, l = self.get_clicks_per_cat_mat(cat)
            student_id[cat] = l
            file_name = self.data_dir + "/" + cat + ".csv"
            print('Writing number of clicks data into csv file ' + file_name)
            with open(file_name, 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(mat)
                f.close()

        return student_id

    def boxplot_clicks_by_cat(self, cat, increase, decrease, noch, filename):
        stat_anova, p_anova = stats.f_oneway(np.arctanh(increase), np.arctanh(decrease), np.arctanh(noch))
        stat_kw, p_kw = stats.kruskal(increase, decrease, noch)
        y = np.array([increase, decrease, noch])
        # Plot group means
        fig = plt.figure(figsize=(8,5))
        plt.boxplot(y, whiskerprops=dict(color='grey', linewidth=1),
                       boxprops=dict(linewidth=2, color='b'),
                       medianprops=dict(color='r', linewidth=3),
                       flierprops=dict(color='grey', alpha=0.7, linestyle='--',
                        marker='o', markeredgewidth=0))
        # plt.xlabel('Grade Group'.upper())
        plt.xticks(np.arange(3)+1, ['Increased', 'Decreased', 'No Change'], fontsize=14)
        plt.ylabel('Numerical Final Grade'.upper(), fontsize=14)
        #plt.ylim(-1.1, 1.1)
        plt.tick_params(top='off', bottom='off', labelsize=14)
        # plt.title('Time-Management Score by Grade Group\nKruskal-Wallis H-statistic: %.3f; p-value: %.3f' % (stat_kw, p_kw), fontsize=13)
        plt.title('%s\nH-STATISTIC: %.3f; P-VALUE: %.3f' % (cat.upper(), stat_kw, p_kw), fontsize=14)
        plt.savefig(filename)
        plt.clf()

    def save_csv_data_as_matrix(self, file_name):
        """
        Parameters
        ----------
        file_name : str
        
        Returns
        -------
        None

        """
        print('Writing number of clicks data into csv file ' + file_name)
        cnt_mat = self.get_num_clicks_per_day_mat()
        header = ['roster_random_id'] + ['day'+str(i) for i in range(self.days_limit)]
        ids_col = np.zeros((self.n_students, 1), dtype=np.int32)
        ids_col[:,0] = np.array(self.idx2id, dtype=np.int32)
        cnt_mat = np.concatenate((ids_col, cnt_mat), axis=1)
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            writer.writerows(cnt_mat)
            f.close()

    def save_overall_counts_as_matrix(self, file_name):
        """
        Parameters
        ----------
        file_name : str
        
        Returns
        -------
        None

        """
        print('Writing number of clicks data into csv file ' + file_name)
        cnt_mat = self.get_num_clicks_per_day_mat()
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(cnt_mat)
            f.close()

        return self.idx2id

    def append_to_merged_data_file(self, file_name, increase, decrease, nochange, change_days):
        lines = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            index = header.index('roster_randomid')
            header.append('changepoint_group')
            header.append('changepoint_date')
            header.append('total_clicks')
            lines.append(header)
            row = 0
            added = []
            for line in reader:
                if line[index] == "":
                    student = -1
                else:
                    student = int(float(line[index]))

                added.append(student)

                if added.count(student) > 1:
                    line.append("---")
                elif student in increase:
                    line.append("increase")
                elif student in decrease:
                    line.append("decrease")
                elif student in nochange:
                    line.append("no change")
                elif student == -1:
                    line.append("---")
                else:
                    line.append("---")

                if "no change" not in line and "---" not in line:
                    change = self.first_day + timedelta(days = change_days[row])
                    line.append(change.strftime('%m/%d/%Y'))
                    row += 1
                else:
                    line.append("---")
                
                if student in self.csvdata:
                    clicks = self.get_num_clicks_per_day(self.csvdata[student], type='all')
                    line.append(np.sum(clicks))
                else:
                    line.append("---")
                
                lines.append(line)

        file_name = file_name[0:-4] + "_merged.csv"
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(lines)

    def get_group_for_id(self, student_id, number):
        ids = []
        for i in number:
            ids.append(student_id[i])
            

        return ids

    def normalize_data(self, lines, indices):
        for i in indices:
            lines[1:, i] = preprocessing.scale(lines[1:, i].astype(float))

        return lines

    
    def create_training_data(self, file_name, types):
        lines = []
        student_ids = []
        if types == "total_clicks":
            lines.append(['total_clicks', 'grade', 'GPA'])
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    clicks = self.get_num_clicks_per_day(self.csvdata[student], type='all')
                    line.append(np.sum(clicks))
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0])

        elif types == "categories":
            lines.append(['assignments_clicks', 'discussion_topics_clicks', 'files_clicks', 'pages_clicks', 'grade', 'GPA'])
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    student_ids.append(student)
                    for cat in ['assignments', 'discussion_topics', 'files', 'pages']:
                        clicks = self.get_num_clicks_per_day(self.csvdata[student], type=cat)
                        line.append(np.sum(clicks))
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0,1,2,3])
            
        elif types == "weeks":
            attribute, data = self.get_num_clicks_att(type_data="week_cal")
            header = ["week 1","week 2","week 3","week 4","week 5", "week 6", "week 7", "week 8", "week 9", "week 10", "grade", "GPA"]
            lines.append(header)

            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    clicks = attribute[student]
                    for title in header[0:10]:
                        if title not in clicks:
                            line.append(0)
                        else:
                            line.append(clicks[title])
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0,1,2,3,4,5,6,7,8,9])
            
        elif types == "weeks_time":
            attribute, data = self.get_num_clicks_att(type_data="week_cal")
            header = ["week 1","week 2","week 3","week 4","week 5", "week 6", "week 7", "week 8", "week 9", "week 10", "grade", "GPA"]
            lines.append(header)
        
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    student_ids.append(student)
                    clicks = data[student]
                    times = self.get_time_on_task_per_student(student, 1200, clicks)
                    for title in header[0:10]:
                        if title not in times:
                            line.append(0)
                        else:
                            line.append(times[title])
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0,1,2,3,4,5,6,7,8,9])
            
        elif types == "total_time":
            lines.append(['total_time', 'grade', "GPA"])
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    clicks = {"created_at": self.csvdata[student]["created_at"]}
                    if len(clicks) > 0:
                        times = self.get_time_on_task_per_student(student, 1200, clicks)
                        line.append(times["created_at"])
                        if self.id2grade[student] in ['A+', 'A']:
                            line.append(1.0)
                            line.append(self.g2pnts[self.id2grade[student]])
                        else:
                            line.append(0.0)
                            line.append(self.g2pnts[self.id2grade[student]])

                        lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0])
            
        elif types == "category_time":
            attribute, data = self.get_num_clicks_att(type_data="category")
            lines.append(['assignments_clicks', 'discussion_topics_clicks', 'files_clicks', 'pages_clicks', 'grade', "GPA"])
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    clicks = data[student]
                    if len(clicks) > 0:
                        times = self.get_time_on_task_per_student(student, 1200, clicks)
                        for title in ['assignments', 'discussion_topics', 'files', 'pages']:
                            if title in times:
                                line.append(times[title])
                            else:
                                line.append(0)
                        if self.id2grade[student] in ['A+', 'A']:
                            line.append(1.0)
                            line.append(self.g2pnts[self.id2grade[student]])
                        else:
                            line.append(0.0)
                            line.append(self.g2pnts[self.id2grade[student]])

                        lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0,1,2,3])
            
        elif types == "total_sessions":
            lines.append(["Number of sessions", "grade", "GPA"])
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    sessions = self.get_number_session(student, "total")
                    line.append(sessions)
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0])
            
        elif types == "week_sessions":
            header = ["week 1","week 2","week 3","week 4","week 5", "week 6", "week 7", "week 8", "week 9", "week 10", "grade", "GPA"]
            lines.append(header)
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    sessions = self.get_number_session(student, "week")
                    for title in header[0:10]:
                        if title not in sessions:
                            line.append(0)
                        else:
                            line.append(sessions[title])
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line)     
            lines = np.array(lines)
            lines = self.normalize_data(lines, [0,1,2,3,4,5,6,7,8,9])
            
        elif types == "background":
            header = ["age", "sattotalscore", "hsgpa",
                      "gpacumulative", "istransfer", "ismale", "lowincomeflag",
                      "firstgenerationflag", "isurm", "grade", "GPA"]

            features = ["age", "sattotalscore", "hsgpa",
                      "gpacumulative", "admissionsstatusdetail", "gender",
                        "lowincomeflag", "firstgenerationflag",
                        "eth2009rollupforreporting"]

            lines.append(header)
            ids = self.demo_info_df["roster_randomid"]
            print(type(ids))
            demo = self.demo_info_df
            for student in self.csvdata:
                line = []
                if student in self.id2grade:
                    i = ids[ids == student].index[0]
                    for f in features:
                        if f == "admissionsstatusdetail":
                            if demo[f][i] == "Transfer":
                                line.append(1)
                            else:
                                line.append(0)
                        elif f == "gender":
                            if demo[f][i] == "Male":
                                line.append(1)
                            else:
                                line.append(0)
                        elif f == "lowincomeflag":
                            if demo[f][i] == "Y":
                                line.append(1)
                            else:
                                line.append(0)
                        elif f == "firstgenerationflag":
                            if demo[f][i] == "Y":
                                line.append(1)
                            else:
                                line.append(0)
                        elif f == "eth2009rollupforreporting":
                            if demo[f][i] == "Black" or demo[f][i] == "Hispanic":
                                line.append(1)
                            else:
                                line.append(0)
                        else:
                            if not math.isnan(demo[f][i]):
                                line.append(demo[f][i])
                            else:
                                line.append(0)
                        
                    if self.id2grade[student] in ['A+', 'A']:
                        line.append(1.0)
                        line.append(self.g2pnts[self.id2grade[student]])
                    else:
                        line.append(0.0)
                        line.append(self.g2pnts[self.id2grade[student]])

                    lines.append(line) 
        
        file_name = file_name + "_" + types + "_training.csv"
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(lines)

        return student_ids
            
    @staticmethod
    def get_cats_from_url(url, depth=1):

        if depth == 1:
            if len(url.split('/')) < 6:
                return None
            return url.split('/')[5]
        else:
            raise('ERROR! depth should be integers between 1 and 3!')

    def get_file_name_from_url(self, url):
        fid2file = self.fid2file
        file_id = self.get_file_id_from_url(url)
        try:
            name = fid2file[file_id] if (file_id is not None) else None
        except:
            name = None
        return name

    @staticmethod
    def is_file_url(url, check="all"):
        opts = ["view", "download", "all"]
        assert check in opts, "check must be one of 'view', 'download', 'all'"
        out = False
        if check is "all":
            out = True if ("files" in url and ("module_item_id" in url or "download" in url)) else False
        if check is "view":
            out = True if ("files" in url) and ("module_item_id" in url) else False
        if check is "download":
            out = True if ("files" in url) and ("download" in url) else False
        return out

    @staticmethod
    def get_file_id_from_url(url):
        if CanvasData.is_file_url(url, check="view"):
            out = int(url.split('?module_item_id')[0].split('/')[-1])
        elif CanvasData.is_file_url(url, check="download"):
            out = int(url.split('/download')[0].split('/')[-1])
        else:
            out = None
        return out


