import os
import numpy as np
import pandas as pd
import re
from src.utils import *

def create_institutional_features(student_info, feature_list, id_cols, out_dir, hdf, to_csv=True):
    """
    Extract institutional features for each student in each class from the cleaned student table and save to disk

    Parameters
    ----------
    student_info : Pandas DataFrame
        Student-by-course information (including demographics, prior academic history, survey responses, etc.)

    feature_list : list
        List of feature names to create, e.g., 'isfemale'

    id_cols : list
        Columns of student(-by-course) identifiers

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """
    feat_dem = student_info[id_cols]
    print('Creating institutional features')

    for feat in feature_list:
        if feat in ['age', 'sattotalscore', 'hsgpa', 'gpacumulative']:
            feat_dem[feat] = student_info[feat]
        if feat == 'isfemale':
            feat_dem['isfemale'] = student_info['gender'].map({'Female': 1, 'Male': 0})
        if feat == 'istransfer':
            feat_dem['istransfer'] = np.where(student_info['admissionsstatusdetail']=='Transfer',
                                      1, np.where(student_info['admissionsstatusdetail'].notnull(), 0, np.nan))
        if feat == 'islowincome':
            feat_dem['islowincome'] = student_info['lowincomeflag'].map({'Y': 1, 'N': 0})
        if feat == 'isfirstgen':
            feat_dem['isfirstgen'] = student_info['firstgenerationflag'].map({'Y': 1, 'N': 0})
        if feat == 'ethnicity':
            feat_dem = pd.concat([feat_dem, pd.get_dummies(student_info['eth2009rollupforreporting'])], axis=1)
        if feat == 'isurm':
            feat_dem['isurm'] = student_info['eth2009rollupforreporting'].map({'Black, non-Hispanic': 1, 'Hispanic': 1,
                                                                           'American Indian / Alaskan Native': 1,
                                                                           'Asian / Pacific Islander': 0,
                                                                           'White, non-Hispanic': 0,
                                                                           'International student': 0})
    feat_dem.set_index(id_cols, inplace=True)

    hdf.put('institutional_features', feat_dem)
    print('institutional features saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'institutional_features.csv')
        feat_dem.to_csv(csv_path)
        print(f'institutional features saved to {csv_path}')


def create_click_features(clickstream, course, feature_list, id_cols, out_dir, hdf, to_csv=True, MAX_SECONDS=900,
                          cat_dict=None):
    """
    Extract click-based features for each student in each class from the cleaned clickstream table and save to disk

    Parameters
    ----------
    clickstream : Pandas DataFrame
        Student-by-course-level institutional information (including prior academic history)

    course: Pandas DataFrame
        Course-level information such as duration

    feature_list : list
        List of features to create, e.g., 'time_by_week'

    id_cols : list
        Columns of student(-by-course) identifiers

    MAX_SECONDS : float
        The maximum interaction time of a single click action, used when approximating interaction time from the time
        lag between two adjacent actions

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """
    clicks_srt = clickstream.sort_values(id_cols+['created_at'])

    # Map each action to the corresponding week of the quarter
    course_week_cuts = course.apply(lambda x: pd.date_range(start=x['start_date'], end=x['start_date']+pd.Timedelta(
        x['term_len'], unit='d'), freq='7d').tolist(), axis=1)
    course_week_cuts = dict(zip(course['course_id'], course_week_cuts))
    course_groups = clicks_srt.groupby('course_id')
    clicks_srt['week'] = course_groups['created_at'].apply(lambda x: pd.cut(x, bins=course_week_cuts[x.name],
                                                                            labels=False) + 1).astype('Int64')
    # Calculate the interaction time (in seconds) of each action as the time lag between two adjacent actions
    # If the lag is longer than MAX_SECONDS or if the action is the last one of a student, set interaction time to
    # MAX_SECONDS
    clicks_srt['interaction_seconds'] = (clicks_srt['created_at'].shift(-1) - clicks_srt['created_at']).dt.total_seconds()
    is_last_action = (clicks_srt['roster_randomid'] != clicks_srt.shift(-1)['roster_randomid'])
    is_too_long = (clicks_srt['interaction_seconds'] > MAX_SECONDS)
    clicks_srt['interaction_seconds'] = np.where((is_last_action | is_too_long), MAX_SECONDS, clicks_srt['interaction_seconds'])
    clicks_srt = clicks_srt[clicks_srt['week'].notnull()] # Remove clicks outside of the course period

    # Get the category of each click
    clicks_srt['category'] = clicks_srt['url'].apply(get_cat_from_url, args=(cat_dict,))

    # Compute different features at the student-course level
    feat_click = clicks_srt[id_cols].drop_duplicates().reset_index(drop=True)
    print('Creating click features')

    for feat in feature_list:

        if re.search(r'first_\d+_weeks', feat):
            week_count = int(re.findall(r'\d+', feat)[0])
            clicks_base = clicks_srt[clicks_srt['week'].between(1, week_count)]
            suffix = f'_first_{week_count}_wks'
        else:
            clicks_base = clicks_srt.copy()
            suffix = ''

        if 'total_clicks' in feat:
            if 'by_week' in feat:
                df = clicks_base.groupby(id_cols+['week'])['action'].count().reset_index().rename(columns={
                    'action': 'clicks_week_'})
                df = pd.pivot_table(df, values=['clicks_week_'], index=id_cols, columns='week')
                df.columns = [col[0] + str(col[1]) for col in df.columns.values]
            elif 'by_category' in feat:
                df = clicks_base.groupby(id_cols+['category'])['action'].count().reset_index().rename(columns={
                    'action': 'clicks_'})
                df = pd.pivot_table(df, values=['clicks_'], index=id_cols, columns='category')
                df.columns = [col[0] + str(col[1]) for col in df.columns.values]
            else:
                df = clicks_base.groupby(id_cols)['action'].count().to_frame().rename(columns={'action':'total_clicks'})

        elif 'total_time' in feat:
            if 'by_week' in feat:
                df = (clicks_base.groupby(id_cols + ['week'])['interaction_seconds'].sum() / 3600).reset_index().rename(
                    columns={'interaction_seconds': 'time_week_'})
                df = pd.pivot_table(df, values=['time_week_'], index=id_cols, columns='week')
                df.columns = [col[0] + str(col[1]) for col in df.columns.values]
            elif 'by_category' in feat:
                df = (clicks_base.groupby(id_cols + ['category'])['interaction_seconds'].sum() / 3600).reset_index(
                      ).rename(columns={'interaction_seconds': 'time_'})
                df = pd.pivot_table(df, values=['time_'], index=id_cols, columns='category')
                df.columns = [col[0] + str(col[1]) for col in df.columns.values]
            else:
                df = (clicks_base.groupby(id_cols)['interaction_seconds'].sum() / 3600).to_frame().rename(columns={
                    'interaction_seconds': 'total_time'})

        df = df.add_suffix(suffix).reset_index()
        feat_click = feat_click.merge(df, how='left')

    feat_click.set_index(id_cols, inplace=True)
    feat_click[feat_click.notnull().any(axis=1)] = feat_click.fillna(0)

    hdf.put('click_features', feat_click)
    print('Click features saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'click_features.csv')
        feat_click.to_csv(csv_path)
        print(f'Click features saved to {csv_path}')


def create_survey_features(student_info, feature_list, id_cols, out_dir, hdf, to_csv=True):
    """
    Extract survey-based features for each student in each class from the cleaned student table and save to disk

    Parameters
    ----------
    student_info : Pandas DataFrame
        Student-by-course information (including demographics, prior academic history, survey responses, etc.)

    feature_list : list
        List of features to create, e.g., 'effort_regulation'

    id_cols : list
        Columns of student(-by-course) identifiers

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """
    feat_svy = student_info[id_cols]
    print('Creating survey features')

    for feat in feature_list:
        if feat == 'effort_regulation':
            feat_svy['effort_regulation'] = (student_info[['pre_er2', 'pre_er4']].join(6 - student_info[['pre_er1',
                                                                                                         'pre_er3']])).mean(axis=1)
        if feat == 'time_management':
            feat_svy['time_management'] = student_info[['pre_orsh4', 'pre_orsh5']].mean(axis=1)
        if feat == 'self_efficacy':
            feat_svy['self_efficacy'] = student_info[['pre_se1', 'pre_se2', 'pre_se3']].mean(axis=1)
        if 'environment_management' in feature_list:
            feat_svy['environment_management'] = student_info[['pre_orsh1', 'pre_orsh2']].mean(axis=1)

    feat_svy.set_index(id_cols, inplace=True)

    hdf.put('survey_features', feat_svy)
    print('Survey features saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'survey_features.csv')
        feat_svy.to_csv(csv_path)
        print(f'Survey features saved to {csv_path}')


def create_labels(student_info, enrollment, label_dict, id_cols, out_dir, hdf, to_csv=True):
    """
    Construct labels of academic performance (within-course and follow-up) and save to disk

    Parameters
    ----------
    student_info : Pandas DataFrame
        Student-by-course information (including demographics, prior academic history, survey responses, etc.)

    enrollment : Pandas DataFrame
        Student-by-course enrollment records (with grades)

    label_dict : dict
        List of features to create, e.g., 'current_current_course_over_median'

    id_cols : list
        Columns of student(-by-course) identifiers

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """

    # From the enrollment history of one single student, derive the (weighted) average GPA
    # Courses without a letter grade (e.g., P/NP) are not calculated
    def get_avg_gpa(df_stud_enroll):
        df_stud_enroll_valid = df_stud_enroll[df_stud_enroll['grade_point'].notnull()]
        if len(df_stud_enroll_valid) == 0:
            return np.nan
        else:
            return np.average(df_stud_enroll_valid['grade_point'], weights=df_stud_enroll_valid['courseunits'])

    labels = student_info[id_cols]
    print('Creating labels')

    for scope in label_dict:
        if scope == 'cur_course':
            for threshold in label_dict[scope]:
                if threshold == 'median':
                    labels['cur_course_over_median'] = student_info.groupby('course_id')['course_total'].transform(
                        lambda x: (x >= x.median()).astype(int))
                    labels['cur_course_over_median'] = np.where(student_info['course_total'].notnull(),
                                                            labels['cur_course_over_median'], np.nan)
                elif convert_letter_to_grade_point(threshold) is not None:
                    labels[f'cur_course_over_{threshold}'] = np.where(student_info['grade'].notnull(), (student_info[
                        'grade_point'] >= convert_letter_to_grade_point(threshold)).astype(int), np.nan)

        if scope == 'next_year':
            next_terms = student_info[['acadyr', 'acadterm']].drop_duplicates()
            next_terms = pd.concat(next_terms.apply(lambda x: get_next_terms(x['acadyr'], x['acadterm'], N_TERMS=4),
                                               axis=1).tolist()).reset_index(drop=True)
            stud_next_year = student_info[id_cols+['acadyr', 'acadterm']].merge(next_terms, how='left').merge(
                enrollment, left_on=['roster_randomid', 'next_acadyr', 'next_acadterm'], right_on=[
                    'roster_randomid', 'acadyr', 'acadterm'], how='left')
            stud_next_year['grade_point'] = stud_next_year['grade'].apply(convert_letter_to_grade_point)
            stud_next_year = stud_next_year.groupby(id_cols).apply(get_avg_gpa).reset_index().rename(columns={0:
                                                                                                         'next_year_GPA'})
            for threshold in label_dict[scope]:
                if threshold == 'median':
                    stud_next_year['next_year_over_median'] = stud_next_year.groupby('course_id')[
                        'next_year_GPA'].transform(lambda x: (x >= x.median()).astype(int))
                    stud_next_year['next_year_over_median'] = np.where(stud_next_year['next_year_GPA'].notnull(),
                                                                       stud_next_year['next_year_over_median'], np.nan)
                elif is_valid_gpa(threshold):
                    stud_next_year[f'next_year_over_{threshold}'] = np.where(stud_next_year['next_year_GPA'].notnull(),
                        (stud_next_year['next_year_GPA'] >= float(threshold)).astype(int), np.nan)
                labels = labels.merge(stud_next_year.drop(columns='next_year_GPA'), how='left')

    labels.set_index(id_cols, inplace=True)

    hdf.put('labels', labels)
    print('Labels saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'labels.csv')
        labels.to_csv(csv_path)
        print(f'Labels saved to {csv_path}')


def create_protected_attributes(student_info, attr_list, id_cols, out_dir, hdf, to_csv=True):
    """
    Construct protected attributes and save to disk

    Parameters
    ----------
    student_info : Pandas DataFrame
        Student-by-course information (including demographics, prior academic history, survey responses, etc.)

    attr_list : list
        List of protected attributes to create

    id_cols : list
        Columns of student(-by-course) identifiers

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    None
    """
    attrs_protected = student_info[id_cols]
    for attr in attr_list:
        if attr == 'ethnicity':
            attrs_protected['ethnicity'] = student_info['eth2009rollupforreporting']
        elif attr == 'gender':
            attrs_protected['gender'] = student_info['gender']
        elif attr == 'low_income':
            attrs_protected['low_income'] = student_info['lowincomeflag'].map({'Y': 'Low-income',
                                                                               'N': 'Not Low-Income'})
        elif attr == 'first_generation':
            attrs_protected['first_gen'] = student_info['firstgenerationflag'].map({'Y': 'First-Gen',
                                                                                    'N': 'Non First-Gen'})
        elif attr == 'hs_achievement':
            attrs_protected['hs_gpa'] = student_info.groupby('course_id')['hsgpa'].transform(pd.qcut, q=4,
                                                                          labels=[f'hsGPA: Q{i+1}' for i in range(4)])
            attrs_protected['hs_gpa'] = attrs_protected['hs_gpa'].astype(str).replace('nan', np.nan)
    attrs_protected.set_index(id_cols, inplace=True)

    hdf.put('protected_attributes', attrs_protected)
    print('Protected attributes saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'protected_attributes.csv')
        attrs_protected.to_csv(csv_path)
        print(f'Protected attributes saved to {csv_path}')


def run(semantic_dir, feature_dir, feature_config):
    """
    Generate features (predictors) and labels (outcomes) from semantic (entity-level) tables, and store them in: 1)
    separate .csv files; and 2) a single HDFStore object

    Parameters
    ----------
    semantic_dir : str
        Directory to store semantic tables

    feature_dir : str
        Directory where extracted features are stored

    feature_config : str
        Name of feature configuration file, with full path

    Returns
    -------
    None
    """

    config = load_yaml(feature_config)
    id_cols = ['course_id', 'roster_randomid']

    with pd.HDFStore(os.path.join(semantic_dir, 'semantic.h5')) as hdf_semantic:
        course = hdf_semantic['course']
        student_info = hdf_semantic['student']
        clickstream = hdf_semantic['click']
        enrollment = hdf_semantic['course_enrolled']

    with pd.HDFStore(os.path.join(feature_dir, 'feature.h5')) as hdf_feature:
        create_institutional_features(student_info, config.get('institutional'), id_cols, feature_dir, hdf_feature)
        create_click_features(clickstream, course, config.get('click'), id_cols, feature_dir, hdf_feature,
                              cat_dict=config.get('url_cats'))
        create_survey_features(student_info, config.get('survey'), id_cols, feature_dir, hdf_feature)
        create_labels(student_info, enrollment, config.get('label'), id_cols, feature_dir, hdf_feature)
        create_protected_attributes(student_info, config.get('protected'), id_cols,
                                    feature_dir, hdf_feature)