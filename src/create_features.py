import os
import numpy as np
import pandas as pd
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
        x['term_len'],unit='d'), freq='7d').tolist(), axis=1)
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
        if feat == 'total_clicks':
            tc = clicks_srt.groupby(id_cols)['action'].count().reset_index().rename(columns={'action': 'total_clicks'})
            feat_click = feat_click.merge(tc, how='left')

        if feat == 'total_clicks_by_week':
            tcw = clicks_srt.groupby(id_cols+['week'])['action'].count().reset_index().rename(columns={'action':
                                                                                                           'clicks_week_'})
            tcw = pd.pivot_table(tcw, values=['clicks_week_'], index=id_cols, columns='week').reset_index()
            tcw.columns = [col[0] + str(col[1]) for col in tcw.columns.values]
            feat_click = feat_click.merge(tcw, how='left')

        if feat == 'total_clicks_first_two_wks':
            tc2w = clicks_srt[clicks_srt['week']<=2].groupby(id_cols)['action'].count().reset_index().rename(columns={
                'action': 'clicks_first_two_wks'})
            feat_click = feat_click.merge(tc2w, how='left')

        if feat == 'total_clicks_by_category':
            tcc = clicks_srt.groupby(id_cols+['category'])['action'].count().reset_index().rename(columns={'action':
                                                                                                               'clicks_'})
            tcc = pd.pivot_table(tcc, values=['clicks_'], index=id_cols, columns='category').reset_index()
            tcc.columns = [col[0] + str(col[1]) for col in tcc.columns.values]
            feat_click = feat_click.merge(tcc, how='left')

        if feat == 'total_time':
            tt = clicks_srt.groupby(id_cols)['interaction_seconds'].sum().reset_index().rename(columns={
                'interaction_seconds': 'total_time'})
            feat_click = feat_click.merge(tt, how='left')

        if feat == 'total_time_by_week':
            ttw = clicks_srt.groupby(id_cols + ['week'])['interaction_seconds'].sum().reset_index().rename(columns={
                'interaction_seconds': 'time_week_'})
            ttw = pd.pivot_table(ttw, values=['time_week_'], index=id_cols, columns='week').reset_index()
            ttw.columns = [col[0] + str(col[1]) for col in ttw.columns.values]
            feat_click = feat_click.merge(ttw, how='left')

        if feat == 'total_time_first_two_wks':
            tt2w = clicks_srt[clicks_srt['week'] <= 2].groupby(id_cols)['interaction_seconds'].sum().reset_index().rename(
                columns={'interaction_seconds': 'time_first_two_wks'})
            feat_click = feat_click.merge(tt2w, how='left')

        if feat == 'total_time_by_category':
            ttc = clicks_srt.groupby(id_cols + ['category'])['interaction_seconds'].sum().reset_index().rename(columns={
                'interaction_seconds': 'time_'})
            ttc = pd.pivot_table(ttc, values=['time_'], index=id_cols, columns='category').reset_index()
            ttc.columns = [col[0] + str(col[1]) for col in ttc.columns.values]
            feat_click = feat_click.merge(ttc, how='left')

    feat_click.set_index(id_cols, inplace=True)

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
            feat_svy['effort_regulation'] = student_info[['pre_er1', 'pre_er2', 'pre_er3', 'pre_er4']].sum(axis=1,
                                                                                                           min_count=1)
        if feat == 'time_management':
            feat_svy['time_management'] = student_info[['pre_orsh4', 'pre_orsh5']].sum(axis=1, min_count=1)
        if feat == 'self_efficacy':
            feat_svy['self_efficacy'] = student_info[['pre_se1', 'pre_se2', 'pre_se3']].sum(axis=1, min_count=1)
        if 'environment_management' in feature_list:
            feat_svy['environment_management'] = student_info[['pre_orsh1', 'pre_orsh2']].sum(axis=1, min_count=1)

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
        df_stud_enroll_valid = df_stud_enroll[df_stud_enroll['GPA'].notnull()]
        if len(df_stud_enroll_valid) == 0:
            return np.nan
        else:
            return np.average(df_stud_enroll_valid['GPA'], weights=df_stud_enroll_valid['courseunits'])

    labels = student_info[id_cols]
    print('Creating labels')

    for scope in label_dict['scope']:
        if scope == 'cur_course':
            for threshold in label_dict['threshold']:
                if threshold == 'median':
                    labels['cur_course_over_median'] = student_info.groupby('course_id')['course_total'].transform(
                        lambda x: (x >= x.median()).astype(int))
                    labels['cur_course_over_median'] = np.where(student_info['course_total'].notnull(),
                                                            labels['cur_course_over_median'], np.nan)
                else:
                    labels[f'cur_course_over_{threshold}'] = student_info.groupby('course_id')['grade'].transform(
                        lambda x: (x >= threshold).astype(int))
                    labels[f'cur_course_over_{threshold}'] = np.where(student_info['grade'].notnull(),
                                                                  labels[f'cur_course_over_{threshold}'], np.nan)
        if scope == 'next_year':
            next_terms = student_info[['acadyr', 'acadterm']].drop_duplicates()
            next_terms = pd.concat(next_terms.apply(lambda x: get_next_terms(x['acadyr'], x['acadterm'], N_TERMS=4),
                                               axis=1).tolist()).reset_index(drop=True)
            stud_next_year = student_info[id_cols+['acadyr', 'acadterm']].merge(next_terms, how='left').merge(
                enrollment, left_on=['roster_randomid', 'next_acadyr', 'next_acadterm'], right_on=[
                    'roster_randomid', 'acadyr', 'acadterm'], how='left')
            stud_next_year['GPA'] = stud_next_year['grade'].apply(convert_letter_to_gpa)
            stud_next_year = stud_next_year.groupby(id_cols).apply(get_avg_gpa).reset_index().rename(columns={0:
                                                                                                         'next_year_GPA'})
            for threshold in label_dict['threshold']:
                if threshold == 'median':
                    stud_next_year['next_year_over_median'] = stud_next_year.groupby('course_id')[
                        'next_year_GPA'].transform(lambda x: (x >= x.median()).astype(int))
                    stud_next_year['next_year_over_median'] = np.where(stud_next_year['next_year_GPA'].notnull(),
                                                                       stud_next_year['next_year_over_median'], np.nan)
                    labels = labels.merge(stud_next_year.drop(columns='next_year_GPA'), how='left')
    labels.set_index(id_cols, inplace=True)

    hdf.put('labels', labels)
    print('Labels saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'labels.csv')
        labels.to_csv(csv_path)
        print(f'Labels saved to {csv_path}')


def create_protected_attributes(student_info, clickstream, enrollment, attr_list, id_cols, out_dir, hdf, to_csv=True):
    """
    Construct protected attributes and save to disk

    Parameters
    ----------
    student_info : Pandas DataFrame
        Student-by-course information (including demographics, prior academic history, survey responses, etc.)

    clickstream : Pandas DataFrame
        Student-by-course-level institutional information (including prior academic history)

    enrollment : Pandas DataFrame
        Student-by-course enrollment records (with grades)

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
            attrs_protected['hs_gpa'] = pd.qcut(student_info['hsgpa'], 4,
                                                labels=[f'hsGPA: Q{i+1}' for i in range(4)]).astype(str)
        elif attr == 'current_achievement':
            attrs_protected['cum_gpa'] = pd.qcut(student_info['gpacumulative'], 4,
                                                 labels=[f'cumGPA: Q{i+1}' for i in range(4)]).astype(str)
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
        create_protected_attributes(student_info, clickstream, enrollment, config.get('protected'), id_cols,
                                    feature_dir, hdf_feature)