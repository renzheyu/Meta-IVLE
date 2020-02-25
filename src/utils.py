###########################
# Commonly used functions #
###########################
import yaml
import numpy as np
import pandas as pd
import re
from itertools import product

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
        Academic year in the form of 20XX-YY (YY=XX+1)
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
        Academic year in the form of 20XX-YY (YY=XX+1)
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

def get_cat_from_url(url, cat_dict=None):
    """
    Get the category of a URL.
    If cat_dict is None, return the raw category as specified in the middle of the URL
        Ex.
        1. https://canvas.eee.uci.edu/courses/2230/files/742190/download -> 'files'
        2. https://canvas.eee.uci.edu/courses/2230/ -> 'homepage'
    If cat_dict is given, map the raw category to the user-specified category in cat_dict
        Ex. 'files' -> 'content'

    Parameters
    ----------
    url : str

    cat_dict: dict
        Mapping of user-specified categories to raw categories
        Ex. {'portal': ['homepage', 'front_page']}

    Returns
    -------
    cat : str
    """
    str_match = re.findall(r'(?<=courses/)\d+/\w+', url)
    if len(str_match)==0:
        cat = 'homepage'
    else:
        cat = re.sub(r'\d+/', '', str_match[0])
    if cat_dict is not None:
        cat_macro = 'misc'
        for k in cat_dict:
            if cat in cat_dict[k]:
                cat_macro = k
                break
    return cat_macro

def get_next_terms(acadyr, acadterm, N_TERMS=4):
    """
    Given one academic term (quarter), find N_TERMS consecutive terms that follow.
        Ex. 2016-17 Spring -> 2016-17 Summer, 2017-18 Fall, 2017-18 Winter, 2017-18 Spring, ...

    Parameters
    ----------
    acadyr : str
        Academic year to start with in the form of 20XX-YY (YY=XX+1)

    acadterm : str
        Academic term name to start with in capitalized, full format, e.g., 'Winter', 'Summer 1'

    N_TERMS : int
        Number of following terms to return

    Returns
    -------
    next_terms : DataFrame
        Each row is a term, e.g.,
            acadyr  | acadterm | next_acadyr  | next_acadterm
            +++++++++++++++++++++++++++++++++++++++++++++++++
            2016-17 | Fall     | 2016-17      | Winter
            2016-17 | Fall     | 2016-17      | Spring
    """
    next_acadyrs = []
    next_acadterms = []
    terms = ['Fall', 'Winter', 'Spring', 'Summer']
    for i, term in enumerate(terms):
        if term in acadterm:
            term_start_index = i
            break

    for j in range(4):
        acadterm_offset = (term_start_index + j + 1) % len(terms)
        next_acadterm = terms[acadterm_offset]
        acadyear_offset = int((term_start_index + j + 1) / len(terms))
        next_acadyr = str(int(acadyr[:4]) + acadyear_offset) + '-' + str(int(acadyr[-2:]) + acadyear_offset)
        next_acadterms.append(next_acadterm)
        next_acadyrs.append(next_acadyr)

    next_terms = pd.DataFrame({'acadyr': [acadyr] * N_TERMS,
                               'acadterm': [acadterm] * N_TERMS,
                               'next_acadyr': next_acadyrs,
                               'next_acadterm': next_acadterms})
    return next_terms

def split_bool_cols(df):
    """
    Separate boolean and non-boolean numeric columns in a Pandas DataFrame
    Useful when all numeric columns are of "float" data type and it is not feasible to separate the columns by data type

    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    bool_cols : list
    non_bool_cols : list
    """
    df_num = df.select_dtypes('number')
    bool_cols = [col for col in df_num.columns if np.isin(df_num[col].dropna().unique(), [0, 1]).all()]
    non_bool_cols = df_num.columns.drop(bool_cols).tolist()
    return bool_cols, non_bool_cols

def get_pred_eval_hits(metric_name, y_true, y_pred):
    """
    Evaluate prediction results of a binary classification given a metric name

    Parameters
    ----------
    metric_name : string

    y_true : numpy array
        True values of the target

    y_pred : numpy array
        Predicted values (0 or 1) of the target

    Returns
    -------
    hits : numpy array or Pandas Series
        Binary indicators of whether each prediction should be regarded as "correct" given the metric
    """
    if metric_name == 'acc':
        hits = np.where(y_true == y_pred, 1, 0)
    elif metric_name == 'fpr':
        hits = np.where(y_true == 0,
                        np.where(y_pred ==1, 1, 0),
                        np.nan)
    elif metric_name == 'fnr':
        hits = np.where(y_true == 1,
                        np.where(y_pred == 0, 1, 0),
                        np.nan)
    return hits

def categorize_table(df):
    """
    Convert all string columns of the existing table into category columns in order to preserve order

    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    df_cat : Pandas DataFrame
    """
    df_cat = df.copy()
    for col in df_cat.select_dtypes(include=object).columns:
        cat_type = pd.CategoricalDtype(categories=df_cat[col].unique(), ordered=True)
        df_cat[col] = df[col].astype(cat_type)
    return df_cat