import numpy as np
import pandas as pd
from src.utils import *


def create_course_table(course_list, term_start_date, out_dir, hdf, to_csv=True):
    """
    Create course-level meta data from config files (currently only including starting dates and durations in days)

    Parameters
    ----------
    course_list : list
        List of course folders (with full path)

    term_start_date : dict
        Dictionary that maps each academic quarter to its starting date

    out_dir : str
        Directory to save the cleaned meta data

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """

    course_table_val = [parse_course_name(course.split('\\')[-1]) for course in course_list]
    course_table = pd.DataFrame(course_table_val)
    course_table['course_id'] = np.arange(len(course_table)) + 1
    course_table['acadyr'] = course_table.apply(lambda x: get_acad_year(x['year'], x['acadterm']), axis=1)
    course_table['start_date'] = course_table.apply(lambda x: term_start_date[x['acadterm'] + ' ' + str(x['year'])],
                                                    axis=1)
    course_table['start_date'] = pd.to_datetime(course_table['start_date']).dt.tz_localize('US/Pacific')
    course_table['term_len'] = course_table['acadterm'].apply(lambda x: 35 if 'Summer' in x else 70)

    hdf.put('course', course_table)
    print('Course table saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'course.csv')
        course_table.to_csv(csv_path, index=False)
        print(f'Course table saved to {csv_path}')


def clean_student_table(course_name, course_id, student_table, col_dict):
    """
    Clean student information table from a given course
    The cleaning process does the following:
    - Change the data type of selected variables
    - Change selected variable names

    Parameters
    ----------
    course_name : str
        Name of the course, in the form of '16S1 PHY 3A'

    course_id : int
        Unique course identifier

    student_table : Pandas Dataframe

    col_dict : dict
        Columns (variables) to be included along with their "thesaurus" for different course, if any, in the form of
        {'varname1': {'coursename1': 'alias1'}}

    Returns
    -------
    cleaned_student_table: Pandas Dataframe
    """

    # TODO: Better organize this temporary filter
    if course_name == '16Fa CHEM 1P':
        student_table = student_table[student_table['coursecode'].astype(int) == 40130]
    course_info_cols = parse_course_name(course_name)
    cleaned_student_table = pd.DataFrame([], index=range(student_table.shape[0]))

    cleaned_student_table['course_id'] = course_id
    for col in col_dict:
        if col in course_info_cols:
            cleaned_student_table[col] = course_info_cols[col]
        elif col == 'acadyr':
            cleaned_student_table[col] = get_acad_year(course_info_cols['year'], course_info_cols['acadterm'])
        elif col_dict[col] is not None and course_name in col_dict[col]:
            cleaned_student_table[col] = student_table[col_dict[col][course_name]]
        elif col in student_table.columns:
            cleaned_student_table[col] = student_table[col]

    # Encode survey responses where necessary
    is_svy_col = cleaned_student_table.columns.str.contains('pre')
    if is_svy_col.sum() > 0:
        svy_cols = cleaned_student_table.columns[is_svy_col]
        for col in svy_cols:
            if cleaned_student_table[col].dtype == 'O':
                cleaned_student_table[col] = cleaned_student_table[col].str.lower()
                cleaned_student_table[col].replace(to_replace={'not at all true of me': 1,
                                                    'very untrue of me': 1,
                                                    'somewhat untrue of me': 2,
                                                    'neutral': 3,
                                                    'very true of me': 5,
                                                    'strongly disagree': 1,
                                                    'strongly agree': 5}, inplace=True)
                if 'neutral' in cleaned_student_table[col].tolist():
                    cleaned_student_table[col].replace('somewhat true of me', 4, inplace=True)
                else:
                    cleaned_student_table[col].replace('somewhat true of me', 3, inplace=True)
            cleaned_student_table[col] = cleaned_student_table[col].astype(float)

    # Set 'coursecode' to string
    if 'coursecode' in cleaned_student_table.columns:
        course_codes = cleaned_student_table['coursecode']
        course_code = course_codes[course_codes.first_valid_index()]
        cleaned_student_table['coursecode'] = str(int(course_code))

    # Calculate letter grades where missing and order
    if 'grade' in cleaned_student_table.columns:
        if 'course_total' in cleaned_student_table.columns:
            grade_to_fill = (cleaned_student_table['course_total'].notnull()) & (cleaned_student_table['grade'].isnull())
            cleaned_student_table['grade'] = np.where(grade_to_fill, cleaned_student_table['course_total'].apply(
                convert_score_to_letter), cleaned_student_table['grade'])
        cleaned_student_table['grade'].astype('category').cat.set_categories(['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+',
                                                                              'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                                                                             ordered=True, inplace=True)

    return cleaned_student_table


def load_student_info(course_dir_list, out_dir, col_dict, hdf, to_csv=True):
    """
    Read and clean student info table for each course and merge across multiple courses
    
    Parameters
    ----------
    course_dir_list : str
        List of course folders (with full path), each of which contains a student-level csv file and a subfolder of
        raw clickstream data
        
    out_dir : str
        Directory to save the cleaned, merged table

    col_dict : dict
        Columns (variables) to be included along with their "thesaurus" in different course contexts, if any,
        in the form of {'varname1': {'coursename1': 'alias1'}}

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination
    
    Returns
    -------
    None
    """
    merged_student_info = pd.DataFrame([])

    for i, course_dir in enumerate(course_dir_list):
        print('Loading student-level information: %s' % course_dir)
        for f in os.listdir(course_dir):
            if f.endswith('csv'):
                student_info = pd.read_csv(os.path.join(course_dir, f))
                break
        print('Finished loading')
        course_name = course_dir.split('\\')[-1]
        course_student_table = clean_student_table(course_name, i+1, student_info, col_dict)
        merged_student_info = merged_student_info.append(course_student_table)
        print('Cleaned data appended')

    merged_student_info['roster_randomid'] = merged_student_info['roster_randomid'].astype('Int64').astype(str)
    merged_student_info.reset_index(inplace=True, drop=True)

    hdf.put('student', merged_student_info)
    print('Merged student info table saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'student.csv')
        merged_student_info.to_csv(csv_path, index=False)
        print(f'Merged student info saved to {csv_path}')


def clean_merge_clicks(course_name, course_id, click_dir):
    """
    Clean and merge clickstream data across students for one course

    Parameters
    ----------
    course_name : str
        Name of the course, in the form of '16S1 PHY 3A'

    course_id : int
        Unique course identifier

    click_dir : str
        Directory where raw clickstream data are saved

    Returns
    -------
    merged_clicks : Pandas DataFrame
        Cleaned merged clickstream data
    """
    click_list = []
    course_info_cols = parse_course_name(course_name)

    for name in os.listdir(click_dir):
        if name.endswith('csv'):
            click_file = pd.read_csv(os.path.join(click_dir, name))
            click_list.append(click_file)
    merged_clicks = pd.concat(click_list).reset_index(drop=True)

    for col in course_info_cols:
        merged_clicks[col] = course_info_cols[col]
    merged_clicks['course_id'] = course_id

    return merged_clicks


def load_clickstream(course_dir_list, out_dir, hdf, to_csv=True):
    """
    Read and clean raw Canvas clickstream data for each course and merge across multiple courses
    
    Parameters
    ----------
    course_dir_list : str
        List of course folders (with full path), each of which contains a student-level csv file and a subfolder of
        raw clickstream data

    out_dir : str
        Directory to save the cleaned, merged table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to store the table in a .csv file (in addition to hdf) for easier examnination

    Returns
    -------
    None
    """

    clickstream_list = []

    for i, course_dir in enumerate(course_dir_list):
        print('Loading clickstream data: %s' % course_dir)
        click_dir = os.path.join(course_dir, 'clickstream')
        course_name = course_dir.split('\\')[-1]
        course_click_table = clean_merge_clicks(course_name, i+1, click_dir)
        clickstream_list.append(course_click_table)
        print('Clickstream appended')

    clickstream = pd.concat(clickstream_list).reset_index(drop=True)
    clickstream['roster_randomid'] = clickstream['roster_randomid'].astype('Int64').astype(str)
    clickstream['created_at'] = pd.to_datetime(clickstream['created_at'], utc=True).dt.tz_convert('US/Pacific')
    clickstream.reset_index(inplace=True, drop=True)

    hdf.put('click', clickstream)
    print('Merged clickstream table saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'click.csv')
        clickstream.to_csv(csv_path, index=False)
        print(f'Merged clickstream saved to {csv_path}')


def load_enrollment(enrollment_file, out_dir, hdf, to_csv=True):
    """
    Read and clean course enrollment history of selected students
    The cleaning process does the following:
    - Change the data type of selected variables
    - Add 'Year' variable
    
    Parameters
    ----------
    enrollment_file : str
        File name with full path
    
    out_dir : str
        Directory to save the cleaned table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to store the table in a .csv file (in addition to hdf) for easier examnination
        
    Returns
    -------
    None
    """

    print('Loading enrollment data: %s' % enrollment_file)
    enrollment = pd.read_csv(enrollment_file)
    print('Loading finished')

    print('Cleaning enrollment data')
    enrollment.columns = enrollment.columns.str.lower()
    enrollment['roster_randomid'] = enrollment['roster_randomid'].astype('Int64').astype(str)
    enrollment['coursecode'] = enrollment['coursecode'].astype(str)
    for int_col in ['acadtermcode', 'ordtermincsmr', 'ordtermexcsmr']:
        enrollment[int_col] = enrollment[int_col].astype(int)
    # enrollment['year'] = enrollment.apply(lambda x: get_year(x['acadyr'], x['acadterm']), axis=1)
    enrollment.reset_index(inplace=True, drop=True)
    print('Cleaning finished')

    hdf.put('course_enrolled', enrollment)
    print('Merged enrollment table saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'course_enrolled.csv')
        enrollment.to_csv(csv_path, index=False)
        print(f'Cleaned data saved to {csv_path}')


def run(raw_data_dir, semantic_dir, data_config):
    """
    Read and organize raw data into semantic (entity-level) tables, stored in: 1) separate .csv files; and 2) a single
    HDFStore object
    
    Parameters
    ----------
    raw_data_dir : str
        Directory where raw data are stored, organized as follows:
        - One csv file with course enrollment history for all students being analyzed
        - Multiple course folders, each containing data for one course      
        
    semantic_dir : str
        Directory to store semantic tables

    data_config : str
        Name of data configuration file, with full path
        
    Returns
    -------
    None
    
    """

    config = load_yaml(data_config)
    course_dir_list = []  # List of course folders

    if 'course_list' in config and 'enrollment_data' in config:
        # If raw data files are explicitly identified
        course_dir_list = [os.path.join(raw_data_dir, course) for course in config['course_list']]
        enrollment_data = os.path.join(raw_data_dir, config['enrollment_data'])
    else:
        # Loop through raw data directory
        for f in os.listdir(raw_data_dir):
            f = os.path.join(raw_data_dir, f)
            if f.endswith('.csv'):
                enrollment_data = f
            elif os.path.isdir(f):
                course_dir_list.append(f)

    with pd.HDFStore(os.path.join(semantic_dir, 'semantic.h5')) as hdf_semantic:
        create_course_table(course_dir_list, config['term_start_date'], semantic_dir, hdf=hdf_semantic)
        load_student_info(course_dir_list, semantic_dir, config['col_dict'], hdf=hdf_semantic) # Student-level tables
        load_clickstream(course_dir_list, semantic_dir, hdf=hdf_semantic)  # Clickstream data
        load_enrollment(enrollment_data, semantic_dir, hdf=hdf_semantic)  # Course enrollment data