###########################
# Commonly used functions #
###########################

import os
import yaml
import pandas as pd
import re

def load_yaml(filename):
    """
    Returns the contents of a yaml file in a list

    Parameters
    ----------
    filename : str
        The full filepath string '.../.../.yaml' of the yaml file to be loaded

    Returns
    -------
    d : dict
        Contents of the yaml file (may be a nested dict)
    """    
    with open(filename, 'r') as ymlfile:
        d = yaml.safe_load(ymlfile)
    return d

def convert_letter_to_gpa(letter, letter_gpa_dict=None):
    """
    Convert letter grade to GPA
    
    Parameters
    ----------
    letter : str
        The letter grade, e.g., 'A+' 
    
    letter_gpa_dict : dict (optional)
        Mapping between letter grade and GPA, in the form of {'letter': GPA}
    
    Returns
    -------
    gpa : float
        Converted GPA on a 4-point scale
    """
    if letter_gpa_dict is None:
        letter_gpa_dict = {'A+':4.0, 'A':4.0, 'A-':3.7, 'B+':3.3, 'B':3.0, 'B-':2.7,
                           'C+':2.3, 'C':2.0, 'C-':1.7, 'D+':1.3, 'D':1.0, 'D-':0.7,
                           'F':0}
    gpa = letter_gpa_dict.get(letter)
    return gpa

def convert_score_to_letter(score, letter_score_dict=None):
    """
    Convert raw score (out of 100) to letter grade based on a set of pre-defined cutoffs, if any

    Parameters
    ----------
    score : float
        The raw score (out of 100)

    letter_score_dict : dict
        Mapping between letter grade and raw score, in the form of {'letter': [lower_bound, upper_bound]}
        where lower bound is inclusive and upper_bound is exclusive. For the highest letter grade, the
        upper bound should be set to 101 to accommodate full mark inputs.

    Returns
    -------
    letter : str
        Converted letter grade (A-F)
    """
    if letter_score_dict is None:
        letter_score_dict = {'A+': [96.5, 101], 'A': [93.5, 96.5], 'A-': [90, 93.5],
                             'B+': [86.5, 90], 'B': [83.5, 86.5], 'B-': [80, 83.5],
                             'C+': [76.5, 80], 'C': [73.5, 76.5], 'C-': [70, 73.5],
                             'D+': [66.5, 70], 'D': [63.5, 66.5], 'D-': [60, 63.5],
                             'F': [0, 60]}
    letter = ''
    for l in letter_score_dict:
        if score >= letter_score_dict[l][0] and score < letter_score_dict[l][1]:
            letter = l
    return letter

def parse_course_name(raw_course_name):
    """
    Parse a course name string into a tuple of course elements

    Parameters
    ----------
    raw_course_name : str
        Course name in the raw data folder, in the customized format of '16S1 PHY 3A'

    Returns
    -------
    course_elem : dict
        Course elements including year, acadterm, dept and coursenum
    """
    m = re.match(r"(\d+)([A-Za-z]+\d?)\s([A-Za-z ]+)\s(\d+[A-Za-z]?)", raw_course_name)

    # year (in four digits)
    year = m.group(1)
    if len(year) == 2:
        year = int('20' + year)

    # academic term (in full)
    acadterm = m.group(2)
    term_dict = {'Wi': 'Winter',
                 'Sp': 'Spring',
                 'S1': 'Summer 1',
                 'S2': 'Summer 2',
                 'Fa': 'Fall'}
    if acadterm in term_dict:
        acadterm = term_dict[acadterm]

    # department offering the course (in short forms)
    dept = m.group(3).upper()

    # course number
    coursenum = m.group(4).upper()

    course_elem = {'year': year, 'acadterm': acadterm, 'dept': dept, 'coursenum': coursenum}
    return course_elem

def get_acad_year(year, term):
    """
    Get the academic year based on absolute year and academic term
    
    Parameters
    ----------
    year : int
        Absolute year, in four digits
    
    term : str
        Academic term name in capitalized, full format, e.g., 'Winter'
        
    Returns
    -------
    acadyear : str
        Academic year in the form of 20XX-XY (Y=X+1)
    """
    if term == 'Fall':
        acadyear = str(year) + '-' + str(year+1)[-2:]
    else:
        acadyear = str(year-1) + '-' + str(year)[-2:]
    return acadyear

def get_year(acadyear, term):
    """
    Get the absolute year based on academic year and academic term
    
    Parameters
    ----------
    acadyear : str
        Academic year in the form of 20XX-XY (Y=X+1)
    term : str
        Academic term name in capitalized, full format, e.g., 'Winter'
    
    Returns
    -------
    year : int
        Absolute year, in four digits
    """
    if term == 'Fall':
        year = int(acadyear[:4])
    else:
        year = int(acadyear[:4]) + 1
    return acadyear

def get_cat_from_url(url):
    """
    Get the category of a URL as specified in the middle of the URL
    Ex.
    1. https://canvas.eee.uci.edu/courses/2230/files/742190/download -> 'files'
    2. https://canvas.eee.uci.edu/courses/2230/ -> 'homepage'

    Parameters
    ----------
    url : str

    Returns
    -------
    cat : str
    """
    if len(url.split('/')) < 6:
        return 'homepage'
    else:
        return url.split('/')[5]

# def generate_quarters(course_students):
#     """
#     Determines the long term quarters
#
#     Returns
#     -------
#     course_students_new: dict
#         keys are courses, values are ids associated with each course including ones with a or b appended to the end
#     """
#     students = dict()
#     course_students_new = dict()
#     for course in course_students:
#         ids = []
#         year = course[:2]
#         quarter = course[2:4]
#         if quarter == "Fa":
#             following = [str(int(year)+1) + "Winter", str(int(year)+1) + "Spring", str(int(year)+1) + "Fall"]
#         elif quarter == "Wi":
#             following = [year + "Spring", year + "Fall", str(int(year)+1)+ "Winter"]
#         elif quarter == "Sp":
#             following = [year + "Fall", str(int(year)+1) + "Winter", str(int(year)+1) + "Spring"]
#         else:
#             following = [year + "Fall", str(int(year)+1) + "Winter", str(int(year)+1) + "Spring"]
#
#         for s in course_students[course]:
#             if str(s) not in students:
#                 students[str(s)] = following
#                 ids.append(str(s))
#             elif str(s) + "a" not in students:
#                 students[str(s) + "a"] = following
#                 ids.append(str(s) + "a")
#             else:
#                 students[str(s) + "b"] = following
#                 ids.append(str(s) + "b")
#         course_students_new[course] = ids
#
#     with open('./student_data.pickle', 'wb') as handle:
#         pickle.dump(students, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     return course_students_new