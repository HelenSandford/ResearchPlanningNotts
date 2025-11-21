import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import plotly.graph_objects as go
import base64 # Import base64 for logo

# Page configuration
st.set_page_config(page_title="Research Planning Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    header {visibility: hidden;}
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .main, .stApp { background-color: #1a1a2e; color: #e8e8e8; }
    h1, h2, h3 { color: #8965e6 !important; }
    [data-testid="stMetricValue"] { color: #03c4da !important; }
    [data-testid="stMetricLabel"] { color: #e8e8e8 !important; }
    [data-testid="stSidebar"] { background-color: #16213e; }
    .stButton>button { background-color: #0f3460; color: #e8e8e8; border: 1px solid #8965e6; }
    .stButton>button:hover { background-color: #8965e6; color: #1a1a2e; }
    input, textarea, .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select,
    .stMultiSelect input, [data-baseweb="input"] input { color: #1a1a1a !important; font-weight: 600 !important; }
    input::placeholder, textarea::placeholder { color: #5d5d5d !important; }
    .stMultiSelect [data-baseweb="tag"], .stMultiSelect span, [data-baseweb="select"] span,
    [data-baseweb="select"] div { color: #1a1a1a !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] button { color: #e8e8e8 !important; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #8965e6 !important; }
    .stFileUploader > label { color: #e8e8e8 !important; }
    [data-testid="stFileUploaderFileName"] { color: #e8e8e8 !important; }
    .stFileUploader button { color: #1a1a1a !important; }
    [data-testid="stMetricLabel"] { color: #e8e8e8 !important; font-weight: 500 !important; }
    .stSelectbox > label, .stMultiSelect > label { color: #e8e8e8 !important; font-weight: 500 !important; }
    .stAlert { background-color: #27DD9E; color: #e8e8e8; }
    .uploaded-file-info { background-color: #2c3e50; padding: 10px; border-radius: 5px; margin: 10px 0; color: #e8e8e8; }
    .streamlit-expanderHeader { color: #e8e8e8 !important; }
    .warning-box { background-color: #ff6b35; color: #fff; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .inherited-rules { background-color: #2c3e50; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #8965e6; }
    /* --- Rule Card Styling to match the box height --- */
    .stMultiSelect, .stNumberInput {
        min-height: 50px; /* Ensure consistency for form elements */
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION CONSTANTS ---
# Organizational Hierarchy Definition
HIERARCHY = {
    'Arts': {
        'CLAS': ['Cultural, Media and Visual Studies', 'Modern Languages and Cultures', 
                 'American and Canadian Studies', 'Cultures, Languages and Area Studies Central'],
        'English': [],
        'Humanities': ['Philosophy', 'History', 'Classics and Archaeology', 'Music', 'Humanities Central']
    },
    'Engineering': {
        'Engineering Central': ['Engineering Central Dept', 'Electrical and Electronic Engineering',
                               'Mechanical Materials and Manuf Eng', 'Chemical and Environmental Engineering',
                               'Architecture and Built Environment', 'Civil Engineering',
                               'Foundation Engineering and Physical Sciences']
    },
    'Science': {
        'Biosciences': ['Food, Nutrition and Dietetics', 'Agricultural and Environmental Sciences',
                       'Plant and Crop Sciences', 'Animal Sciences', 
                       'Microbiology, Brewing and Biotechnology', 'Biosciences Central'],
        'Chemistry': [],
        'Mathematical Sciences': [],
        'Psychology': [],
        'Computer Science': ['Horizon'],
        'Pharmacy': [],
        'Physics and Astronomy': [],
    },
    'Social Sciences': {
        'Geography': [],
        'SSP': [],
        'Economics': [],
        'SPIR': ['Rights Lab'],
        'NUBS': ['HR Business Partnering Policy and Projects'],
        'Education': ['Education Dept', 'Centre for English Language Education'],
        'Law': []
    },
    'Medicine and Health Sciences': {
        'Medicine': ['Life Span and Population Health', 'Mental Health and Clinical Neurosciences',
                    'Translational Medical Sciences', 'Education Centre', 'Nottingham Clinical Trials Unit',
                    'Injury, Recovery and Inflammational Sciences', 'Medicine Central', 
                    'Centre for Health Informatics'],
        'Life Sciences': [],
        'Vet School': [],
        'Health Sciences': []
    }
}

GRADE_OPTIONS = ['RT5*', 'RT6*', 'RT7*', 'Clinical', 
                'RT5-A', 'RT5E-A', 'RT6-A', 'RT6-R', 'RT7-A', 'RT7-R']

# --- INITIALIZATION ---
# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'locked_entities' not in st.session_state:
    st.session_state.locked_entities = {}
if 'selected_entity' not in st.session_state:
    st.session_state.selected_entity = None
if 'selected_level' not in st.session_state:
    st.session_state.selected_level = None
if 'current_criteria' not in st.session_state:
    # Use 'current_criteria' to hold the live, applied rules for the selected unit
    st.session_state.current_criteria = [] 
if 'show_unlock_warning' not in st.session_state:
    st.session_state.show_unlock_warning = False
if 'inherited_from' not in st.session_state:
    st.session_state.inherited_from = None
if 'show_research_groups' not in st.session_state:
    st.session_state.show_research_groups = False
if 'show_rg_analysis' not in st.session_state: 
    st.session_state.show_rg_analysis = False
if 'show_detailed_report' not in st.session_state:
    st.session_state.show_detailed_report = False

# --- UTILITY FUNCTIONS ---

def get_faculty_for_school(school):
    """Get the faculty that contains a given school"""
    for faculty, schools in HIERARCHY.items():
        if school in schools:
            return faculty
    return None

def get_school_for_department(department):
    """Get the school that contains a given department"""
    for faculty, schools in HIERARCHY.items():
        for school, departments in schools.items():
            if department in departments:
                return school
    return None

def get_children(level, entity):
    """Get all children entities for a given level and entity"""
    children = []
    if level == 'Faculty':
        if entity in HIERARCHY:
            for school, departments in HIERARCHY[entity].items():
                children.append(('School', school))
                for dept in departments:
                    children.append(('Department', dept))
    elif level == 'School':
        faculty = get_faculty_for_school(entity)
        if faculty and entity in HIERARCHY[faculty]:
            for dept in HIERARCHY[faculty][entity]:
                children.append(('Department', dept))
    return children

def get_parent_locks(level, entity):
    """Get any parent entity that has locks applied"""
    parents = []
    if level == 'Department':
        school = get_school_for_department(entity)
        if school:
            school_key = f"School::{school}"
            if school_key in st.session_state.locked_entities:
                parents.append(('School', school, st.session_state.locked_entities[school_key]))
            faculty = get_faculty_for_school(school)
            if faculty:
                fac_key = f"Faculty::{faculty}"
                if fac_key in st.session_state.locked_entities:
                    parents.insert(0, ('Faculty', faculty, st.session_state.locked_entities[fac_key]))
    elif level == 'School':
        faculty = get_faculty_for_school(entity)
        if faculty:
            fac_key = f"Faculty::{faculty}"
            if fac_key in st.session_state.locked_entities:
                parents.append(('Faculty', faculty, st.session_state.locked_entities[fac_key]))
    return parents

def get_parent_locks_for_research_group(df, research_group):
    """Check if any staff in the research group have locks from Faculty/School/Department"""
    staff_in_rg = get_staff_for_entity(df, 'Research Group', research_group)
    affected_locks = []
    
    for staff_id in staff_in_rg.index:
        staff_row = df.loc[staff_id]
        if pd.notna(staff_row.get('Faculty')):
            fac_key = f"Faculty::{staff_row['Faculty']}"
            if fac_key in st.session_state.locked_entities:
                affected_locks.append(('Faculty', staff_row['Faculty']))
        if pd.notna(staff_row.get('School')):
            sch_key = f"School::{staff_row['School']}"
            if sch_key in st.session_state.locked_entities:
                affected_locks.append(('School', staff_row['School']))
        if pd.notna(staff_row.get('Department')):
            dept_key = f"Department::{staff_row['Department']}"
            if dept_key in st.session_state.locked_entities:
                affected_locks.append(('Department', staff_row['Department']))
    
    return list(set(affected_locks))

def update_entity_rules(level, entity, criteria):
    """
    Applies the criteria to the current entity and propagates the lock down.
    This replaces lock_entity_and_children for the new auto-lock logic.
    """
    key = f"{level}::{entity}"
    
    if criteria:
        # 1. Apply lock to the current entity
        st.session_state.locked_entities[key] = {
            'criteria': deepcopy(criteria),
            'timestamp': datetime.now().isoformat(),
            'inherited_from': None # This is a direct lock
        }
        
        # 2. Propagate to children (Only for Faculty and School)
        if level in ['Faculty', 'School']:
            children = get_children(level, entity)
            for child_level, child_entity in children:
                child_key = f"{child_level}::{child_entity}"
                
                # Only apply the inherited lock if the child doesn't already have a direct lock
                # We check the 'inherited_from' field to see if the existing lock is a direct lock (None)
                existing_lock = st.session_state.locked_entities.get(child_key)
                
                if existing_lock and existing_lock.get('inherited_from') is None:
                    # Child has a direct lock, do not overwrite or remove it
                    continue
                
                # Apply the inherited lock (or remove an old inherited lock)
                st.session_state.locked_entities[child_key] = {
                    'criteria': deepcopy(criteria),
                    'timestamp': datetime.now().isoformat(),
                    'inherited_from': key # Mark as inherited
                }
    else:
        # If criteria is empty, this means the rules are deleted, so we unlock
        unlock_entity_only(level, entity)

def unlock_entity_only(level, entity):
    """Unlock the entity and all its inherited children"""
    key = f"{level}::{entity}"
    if key in st.session_state.locked_entities:
        del st.session_state.locked_entities[key]
    
    # Also remove all children that inherited from this entity
    children = get_children(level, entity)
    for child_level, child_entity in children:
        child_key = f"{child_level}::{child_entity}"
        if child_key in st.session_state.locked_entities:
            # Only remove if it was inherited from this parent
            if st.session_state.locked_entities[child_key].get('inherited_from') == key:
                del st.session_state.locked_entities[child_key]

def get_entities(data, level):
    entities = set()
    if level == 'Research Group':
        for col in ['Research Group 1', 'Research Group 2', 'Research Group 3', 'Research Group 4']:
            entities.update(data[col].dropna().unique())
    else:
        entities.update(data[level].dropna().unique())
    return sorted(entities)

def get_staff_for_entity(data, level, entity):
    if level == 'Research Group':
        mask = (data['Research Group 1'] == entity) | (data['Research Group 2'] == entity) | \
               (data['Research Group 3'] == entity) | (data['Research Group 4'] == entity)
        return data[mask].copy()
    return data[data[level] == entity].copy()

def matches_grade(grade_name, grade_filter):
    if pd.isna(grade_name):
        return False
    grade_str = str(grade_name)
    if grade_filter == 'RT5*':
        return grade_str.startswith('RT5')
    if grade_filter == 'RT6*':
        return grade_str.startswith('RT6')
    if grade_filter == 'RT7*':
        return grade_str.startswith('RT7')
    if grade_filter == 'Clinical':
        return grade_str.startswith('CL')
    return grade_str == grade_filter

def apply_single_criterion(staff_data, criterion):
    filtered = staff_data.copy()
    
    if criterion.get('grades'):
        grade_mask = pd.Series([False] * len(filtered), index=filtered.index)
        for grade_filter in criterion['grades']:
            grade_mask |= filtered['Grade Name'].apply(lambda x: matches_grade(x, grade_filter))
        filtered = filtered[grade_mask]
    
    if criterion.get('service_years'):
        op, val = criterion['service_years']
        years = filtered['Length of service (years)']
        if op == '>':
            filtered = filtered[years > val]
        elif op == '<':
            filtered = filtered[years < val]
    
    if criterion.get('bottom_percentile') and criterion.get('sort_by'):
        percent = criterion['bottom_percentile']
        sort_metrics = criterion['sort_by']
        
        normalized_df = filtered.copy()
        for metric in sort_metrics:
            if metric in filtered.columns and 'Full-Time Equivalent' in filtered.columns:
                normalized_df[f'{metric}_normalized'] = filtered.apply(
                    lambda row: row[metric] / row['Full-Time Equivalent'] if row['Full-Time Equivalent'] > 0 else 0,
                    axis=1
                )
        
        normalized_cols = [f'{m}_normalized' for m in sort_metrics]
        ascending_list = [True] * len(normalized_cols)
        normalized_df = normalized_df.sort_values(by=normalized_cols, ascending=ascending_list, kind='stable')
        
        total_fte = normalized_df['Full-Time Equivalent'].sum()
        target_fte_to_exclude = total_fte * percent / 100
        
        cumulative_fte = 0
        to_exclude = []
        for idx, row in normalized_df.iterrows():
            if cumulative_fte + row['Full-Time Equivalent'] <= target_fte_to_exclude:
                to_exclude.append(idx)
                cumulative_fte += row['Full-Time Equivalent']
            else:
                break
        
        return set(to_exclude)

    # If not a percentile rule, 'filtered' contains the staff who satisfied the Grade/Service criteria (i.e., staff to KEEP)
    # We must return the set of staff who DID NOT make the cut (i.e., staff to EXCLUDE).
    excluded_ids = set(staff_data.index) - set(filtered.index)
    return excluded_ids

def apply_criteria(staff_data, criteria_list):
    if not criteria_list:
        return set()
    excluded = set()
    for criterion in criteria_list:
        excluded.update(apply_single_criterion(staff_data, criterion))
    return excluded

def get_rt_cost(grade_name):
    """Get the R&T contract cost for a given grade - GROUPED"""
    if pd.isna(grade_name):
        return 0
    grade_str = str(grade_name)
    if grade_str.startswith('RT5'):
        return 51573
    elif grade_str.startswith('RT6'):
        return 69488
    elif grade_str.startswith('RT7'):
        return 102836
    else:
        return 0

def create_metrics_table(before_metrics, after_metrics):
    """Create a standardized metrics comparison table with percentage changes"""
    
    def format_change_with_pct(change_val, before_val):
        """Format change with percentage in brackets"""
        if before_val == 0:
            pct_change = 0
        else:
            pct_change = (change_val / before_val) * 100
            
        if isinstance(change_val, float):
            return f"{change_val:,.2f} ({pct_change:+.1f}%)"
        else:
            return f"{change_val:,} ({pct_change:+.1f}%)"
    
    # Calculate raw differences
    count_change = after_metrics['count'] - before_metrics['count']
    fte_change = after_metrics['fte'] - before_metrics['fte']
    total_coi_change = after_metrics['total_coi'] - before_metrics['total_coi']
    coi_per_fte_change = after_metrics['coi_per_fte'] - before_metrics['coi_per_fte']
    total_schol_change = after_metrics['total_schol'] - before_metrics['total_schol']
    schol_per_fte_change = after_metrics['schol_per_fte'] - before_metrics['schol_per_fte']
    total_cit_change = after_metrics['total_cit'] - before_metrics['total_cit']
    cit_per_fte_change = after_metrics['cit_per_fte'] - before_metrics['cit_per_fte']
    cit_per_pub_change = after_metrics['cit_per_pub'] - before_metrics['cit_per_pub']
    total_rt_cost_change = after_metrics['total_rt_cost'] - before_metrics['total_rt_cost']
    pgr_per_fte_change = after_metrics['pgr_per_fte'] - before_metrics['pgr_per_fte']
    ese_per_fte_change = after_metrics['ese_per_fte'] - before_metrics['ese_per_fte']
    
    metrics_data = {
        'Metric': ['Staff Count', 'Total FTE', 'Total CoI (£)', 'Avg CoI/FTE (£)',
                  'Total Scholarly Output', 'Avg Scholarly/FTE', 'Total Citations',
                  'Avg Citations/FTE', 'Citations per Publication', 'Total R&T Cost (£) ⚠️',
                  'Total Active PGRs', 'PGR per FTE', 'Total ESE Contact Hours', 'ESE Contact Hours per FTE'
                  ],
        'Before': [
            before_metrics['count'], f"{before_metrics['fte']:.2f}", f"£{before_metrics['total_coi']:,.0f}",
            f"£{before_metrics['coi_per_fte']:,.0f}", f"{before_metrics['total_schol']:,.0f}",
            f"{before_metrics['schol_per_fte']:.2f}", f"{before_metrics['total_cit']:,.0f}",
            f"{before_metrics['cit_per_fte']:.0f}", f"{before_metrics['cit_per_pub']:.2f}",
            f"£{before_metrics['total_rt_cost']:,.0f}", f"{before_metrics['total_pgr']:,.0f}",
            f"{before_metrics['pgr_per_fte']:.2f}", f"{before_metrics['total_ese_contact']:,.0f}", 
            f"{before_metrics['ese_per_fte']:,.0f}"
        ],
        'After': [
            after_metrics['count'], f"{after_metrics['fte']:.2f}", f"£{after_metrics['total_coi']:,.0f}",
            f"£{after_metrics['coi_per_fte']:,.0f}", f"{after_metrics['total_schol']:,.0f}",
            f"{after_metrics['schol_per_fte']:.2f}", f"{after_metrics['total_cit']:,.0f}",
            f"{after_metrics['cit_per_fte']:.0f}", f"{after_metrics['cit_per_pub']:.2f}",
            f"£{after_metrics['total_rt_cost']:,.0f}", "-",
            f"{after_metrics['pgr_per_fte']:.2f}", "-", 
            f"{after_metrics['ese_per_fte']:,.0f}"
        ],
        'Change': [
                    format_change_with_pct(count_change, before_metrics['count']),
                    format_change_with_pct(fte_change, before_metrics['fte']),
                    f"£{total_coi_change:,.0f} ({((total_coi_change / before_metrics['total_coi'] * 100) if before_metrics['total_coi'] != 0 else 0):+.1f}%)",
                    f"£{coi_per_fte_change:,.0f} ({((coi_per_fte_change / before_metrics['coi_per_fte'] * 100) if before_metrics['coi_per_fte'] != 0 else 0):+.1f}%)",
                    format_change_with_pct(total_schol_change, before_metrics['total_schol']),
                    format_change_with_pct(schol_per_fte_change, before_metrics['schol_per_fte']),
                    format_change_with_pct(total_cit_change, before_metrics['total_cit']),
                    format_change_with_pct(cit_per_fte_change, before_metrics['cit_per_fte']),
                    format_change_with_pct(cit_per_pub_change, before_metrics['cit_per_pub']),
                    f"£{total_rt_cost_change:,.0f} ({((total_rt_cost_change / before_metrics['total_rt_cost'] * 100) if before_metrics['total_rt_cost'] != 0 else 0):+.1f}%)",
                    "-",  # Total Active PGRs
                    format_change_with_pct(pgr_per_fte_change, before_metrics['pgr_per_fte']),
                    "-",  # Total ESE Contact Hours
                    format_change_with_pct(ese_per_fte_change, before_metrics['ese_per_fte'])
        ]
    }
    return pd.DataFrame(metrics_data)

def calculate_metrics(staff_data, excluded_ids=None):
    """
    Calculates metrics.
    If excluded_ids is None, this calculates the 'Before' metrics.
    If excluded_ids is a set, this calculates the 'After' metrics.
    """
    if excluded_ids is None:
        excluded_ids = set()

    # The staff included in the AFTER calculation are those in staff_data whose IDs are NOT in excluded_ids
    included = staff_data[~staff_data.index.isin(excluded_ids)]
    
    if len(included) == 0:
        return {'count': 0, 'fte': 0, 'total_coi': 0, 'coi_per_fte': 0,
                'total_schol': 0, 'schol_per_fte': 0, 'total_cit': 0,
                'cit_per_fte': 0, 'cit_per_pub': 0, 'total_rt_cost': 0,
                'total_pgr': 0, 'pgr_per_fte': 0, 'total_ese_contact': 0, 'ese_per_fte': 0}
    
    total_fte = included['Full-Time Equivalent'].sum()
    total_coi = included['CoI income (£)'].sum()
    total_schol = included['Scholarly Output'].sum()
    total_cit = included['Citations'].sum()
    
    total_rt_cost = included.apply(lambda row: get_rt_cost(row['Grade Name']) * row['Full-Time Equivalent'], axis=1).sum()

    # For PGR and ESE, the totals should be based on the *original* set of staff (staff_data), 
    # as excluding staff doesn't remove the students/teaching hours from the unit, only the FTE supporting them.
    
    total_pgr = staff_data['PGR Active students'].sum() if 'PGR Active students' in staff_data.columns else 0
    
    total_ese_contact = 0
    if all(col in staff_data.columns for col in ['ESE Hours Timetabled', 'ESE Sessions Timetabled', 'ESE Headcount Timetabled']):
        staff_data_copy = staff_data.copy()
        staff_data_copy['ESE_Contact_Hours'] = (staff_data_copy['ESE Hours Timetabled'] * staff_data_copy['ESE Sessions Timetabled'] * staff_data_copy['ESE Headcount Timetabled'])
        total_ese_contact = staff_data_copy['ESE_Contact_Hours'].sum()
    
    return {
        'count': len(included), 'fte': total_fte, 'total_coi': total_coi,
        'coi_per_fte': total_coi / total_fte if total_fte > 0 else 0,
        'total_schol': total_schol,
        'schol_per_fte': total_schol / total_fte if total_fte > 0 else 0,
        'total_cit': total_cit,
        'cit_per_fte': total_cit / total_fte if total_fte > 0 else 0,
        'cit_per_pub': total_cit / total_schol if total_schol > 0 else 0,
        'total_rt_cost': total_rt_cost,
        'total_pgr': total_pgr,
        'pgr_per_fte': total_pgr / total_fte if total_fte > 0 else 0,
        'total_ese_contact': total_ese_contact,
        'ese_per_fte': total_ese_contact / total_fte if total_fte > 0 else 0
    }

def get_locked_excluded_ids(df):
    """
    Computes the final set of excluded staff IDs based on all active locks.
    Higher priority (closer to staff) locks override lower priority ones.
    """
    staff_exclusions = {}  # staff_id -> (excluded: True/False, lock_key)
    
    # Priority: Faculty (lowest) < School < Department < Research Group (highest)
    priority_map = {'Faculty': 1, 'School': 2, 'Department': 3, 'Research Group': 4}
    
    # Sort all locks by priority (lowest first)
    sorted_locks = sorted(
        st.session_state.locked_entities.items(),
        key=lambda x: priority_map.get(x[0].split('::')[0], 0)
    )
    
    # Process each lock
    for lock_key, lock_data in sorted_locks:
        lock_level, lock_entity = lock_key.split('::')
        
        # 1. Get ALL staff in the entity associated with the lock
        lock_staff = get_staff_for_entity(df, lock_level, lock_entity)
        
        # 2. Identify the staff EXCLUDED by THIS lock's criteria
        lock_excluded = apply_criteria(lock_staff, lock_data['criteria'])
        
        # 3. For all staff in this entity, update their final exclusion status
        for staff_id in lock_staff.index:
            is_excluded_by_this_lock = staff_id in lock_excluded
            current_priority = priority_map.get(lock_level, 0)
            
            if staff_id not in staff_exclusions:
                staff_exclusions[staff_id] = (is_excluded_by_this_lock, lock_key)
            else:
                _, existing_key = staff_exclusions[staff_id]
                existing_level = existing_key.split('::')[0]
                existing_priority = priority_map.get(existing_level, 0)
                
                # If current lock is higher priority, or same priority, overwrite
                if current_priority >= existing_priority:
                    staff_exclusions[staff_id] = (is_excluded_by_this_lock, lock_key)
    
    # Return all staff marked as excluded
    return {staff_id for staff_id, (is_excluded, _) in staff_exclusions.items() if is_excluded}

def get_exclusion_reasons(df, staff_id):
    """Provides a list of lock entities that exclude the staff member."""
    reasons = []
    
    priority_map = {'Faculty': 1, 'School': 2, 'Department': 3, 'Research Group': 4}
    potential_locks = []
    
    if staff_id in df.index:
        staff_row = df.loc[staff_id]
        levels_to_check = ['Faculty', 'School', 'Department']
        
        for level in levels_to_check:
            entity = staff_row.get(level)
            if pd.notna(entity):
                key = f"{level}::{entity}"
                if key in st.session_state.locked_entities:
                    potential_locks.append((key, st.session_state.locked_entities[key]))
        
        for col in ['Research Group 1', 'Research Group 2', 'Research Group 3', 'Research Group 4']:
            rg = staff_row.get(col)
            if pd.notna(rg):
                key = f"Research Group::{rg}"
                if key in st.session_state.locked_entities:
                    potential_locks.append((key, st.session_state.locked_entities[key]))

    potential_locks.sort(key=lambda x: priority_map.get(x[0].split('::')[0], 0))

    final_exclusion_key = None
    
    for lock_key, lock_data in potential_locks:
        lock_level, lock_entity = lock_key.split('::')
        
        lock_staff = get_staff_for_entity(df, lock_level, lock_entity)
        lock_excluded = apply_criteria(lock_staff, lock_data['criteria'])
        
        if staff_id in lock_excluded:
            final_exclusion_key = lock_key
    
    if final_exclusion_key:
        lock_level, lock_entity = final_exclusion_key.split('::')
        lock_data = st.session_state.locked_entities[final_exclusion_key]
        
        inherited = lock_data.get('inherited_from')
        if inherited:
            parent_level, parent_entity = inherited.split('::')
            reasons.append(f"{lock_entity} ({lock_level}) [inherited from {parent_entity}]")
        else:
            reasons.append(f"{lock_entity} ({lock_level})")
            
    return reasons
# --- STREAMLIT UI COMPONENTS ---

# Header row
col_logo, col_title, col_upload = st.columns([2, 3, 2])

# Add a dummy logo.png in base64 if it doesn't exist for running the code
try:
    with col_logo:
        try:
            logo_base64 = base64.b64encode(open("logo.png", "rb").read()).decode()
        except FileNotFoundError:
            logo_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
            
        st.markdown(f"""
            <div style='display: flex; align-items: center;'>
                <img src='data:image/png;base64,{logo_base64}' width='75' style='margin-right: 15px;'/>
                <span style='color: #E8E8E8; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;'>
                    SEA Consultancy Ltd<span style='font-size: 12px; font-weight: 400;'> © 2025</span>
                </span>
            </div>
        """, unsafe_allow_html=True)
except Exception:
    with col_logo:
        st.markdown("## SEA Consultancy Ltd")


with col_title:
    st.title("Research Planning Dashboard")

with col_upload:
    if st.session_state.data is None:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.index.name = 'StaffID' 
                
                numeric_cols = ['Full-Time Equivalent', 'Length of service (years)', 'CoI income (£)',
                              'Nr of research projects', 'Scholarly Output', 'Citations',
                              'Citations per Publication', 'PGR Active students', 
                              'ESE Hours Timetabled', 'ESE Sessions Timetabled', 'ESE Headcount Timetabled']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                for col in ['Faculty', 'School', 'Department']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).replace({'nan': np.nan}) 
                
                st.session_state.data = df
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        st.markdown(f"<div class='uploaded-file-info'>📊 {len(st.session_state.data)} records loaded</div>", unsafe_allow_html=True)
        if st.button("📤 Upload Different File", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'data' and not key.startswith('FormSubmitter'):
                    del st.session_state[key]
            st.session_state.data = None
            st.rerun()

st.markdown("---")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Calculate overall metrics
    locked_excluded_global = get_locked_excluded_ids(df)
    before_global = calculate_metrics(df)
    after_global = calculate_metrics(df, locked_excluded_global)
    
    col_summary, col_main = st.columns([1, 4])
    
    # LEFT: Summary Panel
    with col_summary:
        if st.session_state.selected_entity and st.session_state.selected_level:
            # --- ENTITY SUMMARY ---
            entity = st.session_state.selected_entity
            level = st.session_state.selected_level
            key = f"{level}::{entity}"

            # If the unit has rules defined locally (either directly or inherited)
            if key in st.session_state.locked_entities:
                # The displayed rules are the applied rules
                applied_criteria = st.session_state.locked_entities[key]['criteria']
            else:
                # The unit is completely unlocked/uninherited, applied rules are empty
                applied_criteria = []
                
            # 1. Get ALL staff belonging to the selected entity
            entity_staff_all = get_staff_for_entity(df, level, entity)
            
            # 2. Filter the global exclusion set to only those staff in this entity
            entity_staff_excluded_global = locked_excluded_global.intersection(entity_staff_all.index)
            
            # 3. Calculate 'Before' (all staff in entity)
            entity_before = calculate_metrics(entity_staff_all)
            
            # 4. Calculate 'After' 
            # Recalculate the excluded set if the rules were just edited (which now automatically locks)
            # Since the global lock calculation (get_locked_excluded_ids) already runs before this section, 
            # and rule changes trigger a rerun, we can just rely on the global set.
            entity_after = calculate_metrics(entity_staff_all, entity_staff_excluded_global)
            
            st.markdown(f"### {entity} Summary")
            
            # Calculate deltas for metrics
            staff_delta = entity_after['count'] - entity_before['count']
            fte_delta = entity_after['fte'] - entity_before['fte']
            coi_delta = entity_after['coi_per_fte'] - entity_before['coi_per_fte']
            schol_delta = entity_after['schol_per_fte'] - entity_before['schol_per_fte']
            
            st.metric("Staff", entity_after['count'], 
                     delta=f"{staff_delta:+d}" if staff_delta != 0 else None,
                     delta_color="inverse")
            st.metric("FTE", f"{entity_after['fte']:.1f}",
                     delta=f"{fte_delta:+.1f}" if fte_delta != 0 else None,
                     delta_color="inverse")
            st.metric("CoI/FTE", f"£{entity_after['coi_per_fte']/1000:.0f}k",
                     delta=f"{coi_delta/1000:+.0f}k" if coi_delta != 0 else None,
                     delta_color="normal")
            st.metric("ScOutput/FTE", f"{entity_after['schol_per_fte']:.2f}",
                     delta=f"{schol_delta:+.2f}" if schol_delta != 0 else None,
                     delta_color="normal")
            
            st.markdown("---")
            if st.button("← Back to Overview", use_container_width=True):
                st.session_state.selected_entity = None
                st.session_state.selected_level = None
                st.session_state.current_criteria = [] # Reset to empty list
                st.session_state.show_unlock_warning = False
                st.session_state.inherited_from = None
                st.rerun()
        else:
            # --- OVERALL SUMMARY ---
            st.markdown("### Overall Summary")
            
            staff_delta = after_global['count'] - before_global['count']
            fte_delta = after_global['fte'] - before_global['fte']
            coi_delta = after_global['coi_per_fte'] - before_global['coi_per_fte']
            schol_delta = after_global['schol_per_fte'] - before_global['schol_per_fte']
            
            st.metric("Staff", after_global['count'],
                     delta=f"{staff_delta:+d}" if staff_delta != 0 else None,
                     delta_color="inverse")
            st.metric("FTE", f"{after_global['fte']:.1f}",
                     delta=f"{fte_delta:+.1f}" if fte_delta != 0 else None,
                     delta_color="inverse")
            st.metric("CoI/FTE", f"£{after_global['coi_per_fte']/1000:.0f}k",
                     delta=f"{coi_delta/1000:+.0f}k" if coi_delta != 0 else None,
                     delta_color="normal")
            st.metric("ScOutput/FTE", f"{after_global['schol_per_fte']:.2f}",
                     delta=f"{schol_delta:+.2f}" if schol_delta != 0 else None,
                     delta_color="normal")
            st.metric("Locked Units", len(st.session_state.locked_entities))
            
            st.markdown("---")
            if st.button("Detailed Report", use_container_width=True, type="primary"):
                st.session_state.show_detailed_report = not st.session_state.get('show_detailed_report', False)
                st.rerun()
    
    # RIGHT: Main Content
    with col_main:
        if st.session_state.selected_entity and st.session_state.selected_level:
            entity = st.session_state.selected_entity
            level = st.session_state.selected_level
            key = f"{level}::{entity}"
            
            # --- Rule Management Interface ---
            
            # 1. Determine applied rules and parent status
            is_locked = key in st.session_state.locked_entities
            
            if level == 'Research Group':
                parent_locks_info = get_parent_locks_for_research_group(df, entity)
                has_parent_lock = len(parent_locks_info) > 0
                is_inherited = False
            else:
                parent_locks = get_parent_locks(level, entity)
                has_parent_lock = len(parent_locks) > 0
                is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
            
            st.markdown(f"# {entity}")
            st.caption(f"{level}")
            
            # Show parent lock information (Warning/Info boxes)
            # If the current unit is NOT directly locked, but is affected by a parent lock:
            if not is_locked or is_inherited: 
                if has_parent_lock and level != 'Research Group':
                    st.markdown(f"""
                    <div class='inherited-rules'>
                        <strong>ℹ️ Inherited Rules:</strong> This unit is applying rules from a higher level. Any local rule added here will override parent rules for this unit.
                    </div>
                    """, unsafe_allow_html=True)
                elif has_parent_lock and level == 'Research Group':
                    unique_locks = list(set(parent_locks_info))
                    lock_text = ', '.join([f"{entity} ({lvl})" for lvl, entity in unique_locks])
                    st.markdown(f"""
                    <div class='warning-box'>
                        <strong>⚠️ Warning:</strong> Some staff in this research group have existing rules from: {lock_text}<br>
                        <small>Rules applied to this research group will override previous Faculty/School/Department rules for these staff members.</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # If the unit is directly locked, but is a lower level (School/Department)
            if is_locked and not is_inherited and level in ['School', 'Department']:
                # Show the message that direct rules override parents, and list excluded count
                dept_staff = get_staff_for_entity(df, level, entity)
                # Check staff excluded by the unit's OWN rules
                local_excluded = apply_criteria(dept_staff, st.session_state.locked_entities[key]['criteria'])

                st.markdown(f"""
                <div class='warning-box'>
                    <strong>✅ Active Local Rules.</strong><br>
                    <small>• Rules applied here override any higher-level rules for all staff in this unit.</small><br>
                    <small>• Currently excluding {len(local_excluded)} staff from this unit.</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            st.markdown("### Exclusion Rules")

            # Rules to display are always the applied rules, which are stored in the lock dictionary
            rules_to_display = st.session_state.locked_entities[key]['criteria'] if is_locked else st.session_state.current_criteria

            # --- Rule Editing and Dynamic Add Box ---
            num_rules = len(rules_to_display)
            rules_per_row = 3
            
            # Use a copy of the applied rules to manage changes before re-saving the lock
            st.session_state.current_criteria = deepcopy(rules_to_display) 

            for row_start in range(0, num_rules + 1, rules_per_row):
                # We iterate up to num_rules + 1 to include the slot for the dynamic add box
                cols = st.columns(rules_per_row)
                
                for col_idx, idx in enumerate(range(row_start, min(row_start + rules_per_row, num_rules + 1))):
                    
                    if idx < num_rules:
                        # --- Display existing Rule Card ---
                        with cols[col_idx]:
                            criterion = st.session_state.current_criteria[idx]
                            
                            st.markdown(f"**Rule {idx + 1}**")
                            
                            # Delete button
                            if st.button("Delete", key=f"del_{idx}", use_container_width=True):
                                st.session_state.current_criteria.pop(idx)
                                # Automatically apply the updated rules (re-save the lock)
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()

                            # Grades multiselect
                            selected_grades = st.multiselect("Grades", GRADE_OPTIONS,
                                default=criterion.get('grades', []),
                                key=f"grades_{idx}",
                                label_visibility="collapsed",
                                placeholder="Choose grades or leave blank to select ALL")
                            
                            # Check and save changes
                            if selected_grades != st.session_state.current_criteria[idx].get('grades', []):
                                st.session_state.current_criteria[idx]['grades'] = selected_grades
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()

                            # Service Years
                            st.caption("Years of Service")
                            col_op, col_val = st.columns([1, 2])
                            
                            service_op_default = criterion.get('service_years', ('', 0))[0]
                            service_val_default = float(criterion.get('service_years', ('', 0))[1])
                            
                            with col_op:
                                service_op = st.selectbox("Operator", ['', '>', '<'],
                                    index=['', '>', '<'].index(service_op_default),
                                    key=f"service_op_{idx}",
                                    label_visibility="collapsed")
                            with col_val:
                                service_val = st.number_input("Years", min_value=0.0, step=1.0,
                                    value=service_val_default,
                                    key=f"service_val_{idx}",
                                    label_visibility="collapsed")
                            
                            if service_op and (service_op, service_val) != st.session_state.current_criteria[idx].get('service_years'):
                                st.session_state.current_criteria[idx]['service_years'] = (service_op, service_val)
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()
                            elif not service_op and 'service_years' in st.session_state.current_criteria[idx]:
                                del st.session_state.current_criteria[idx]['service_years']
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()
                            
                            # Bottom %
                            st.caption("Sort by %")
                            percent = st.number_input("Bottom %", min_value=0, max_value=100,
                                value=criterion.get('bottom_percentile', 0),
                                key=f"percent_{idx}",
                                label_visibility="collapsed")
                            
                            if percent != st.session_state.current_criteria[idx].get('bottom_percentile', 0):
                                st.session_state.current_criteria[idx]['bottom_percentile'] = percent
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()
                            
                            # Metrics multiselect
                            metric_options = ['CoI income (£)', 'Scholarly Output', 'Citations']
                            selected_metrics = st.multiselect("Metrics", metric_options,
                                default=criterion.get('sort_by', []),
                                key=f"metrics_{idx}",
                                label_visibility="collapsed",
                                placeholder="Exclude bottom % by... (multiselect in preference order)")
                            
                            if selected_metrics != st.session_state.current_criteria[idx].get('sort_by', []):
                                st.session_state.current_criteria[idx]['sort_by'] = selected_metrics
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()
                            
                            if selected_metrics:
                                st.caption(" → ".join([f"{i+1}. {m.split('(')[0].strip()}" for i, m in enumerate(selected_metrics)]))
                    
                    elif idx == num_rules and (row_start + col_idx) == num_rules:
                        # --- Display Dynamic Add Button ---
                        with cols[col_idx]:
                            st.markdown("### <br>", unsafe_allow_html=True) # Use spacing for vertical alignment
                            if st.button("➕ Add New Rule", key="add_rule_dynamic", use_container_width=True, type="primary"):
                                # Add a new empty rule to the current criteria
                                st.session_state.current_criteria.append({})
                                # Automatically apply the updated rules (re-save the lock)
                                update_entity_rules(level, entity, st.session_state.current_criteria)
                                st.rerun()

            if num_rules == 0:
                st.info("No exclusion rules are currently active for this unit.")
            
            st.markdown("---")
            st.markdown("### Detailed Metrics")

            metrics_table = create_metrics_table(entity_before, entity_after)
            st.dataframe(metrics_table, use_container_width=True, hide_index=True,  height=len(metrics_table) * 35 + 38)
        
        # Show Detailed Report if button clicked
        elif st.session_state.get('show_detailed_report', False):
            st.markdown("### Detailed Metrics Report")
            
            col_back, _ = st.columns([1, 5])
            with col_back:
                if st.button("← Back", key="back_from_report"):
                    st.session_state.show_detailed_report = False
                    st.rerun()

            metrics_table = create_metrics_table(before_global, after_global)
            st.dataframe(metrics_table, use_container_width=True, hide_index=True,  height=len(metrics_table) * 35 + 38)
            
            st.markdown("---")
            st.markdown("### Visual Analysis")
            
            def categorize_grade(grade):
                if pd.isna(grade): return 'Other'
                grade_str = str(grade)
                if grade_str.startswith('RT5'): return 'RT5'
                elif grade_str.startswith('RT6'): return 'RT6'
                elif grade_str.startswith('RT7'): return 'RT7'
                elif grade_str.startswith('CL'): return 'Clinical'
                else: return 'Other'
            
            def categorize_service(years):
                if years < 5: return '0-5'
                elif years < 10: return '5-10'
                elif years < 15: return '10-15'
                elif years < 20: return '15-20'
                elif years < 25: return '20-25'
                elif years < 30: return '25-30'
                else: return '30+'
            
            df_before = df.copy()
            df_after = df[~df.index.isin(locked_excluded_global)].copy()
            
            # Chart 1: FTE by Grade Type
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### FTE by Grade Type")
                df_before['Grade_Category'] = df_before['Grade Name'].apply(categorize_grade)
                df_after['Grade_Category'] = df_after['Grade Name'].apply(categorize_grade)
                
                grade_order = ['RT5', 'RT6', 'RT7', 'Clinical', 'Other']
                grade_before = df_before.groupby('Grade_Category')['Full-Time Equivalent'].sum().reindex(grade_order, fill_value=0)
                grade_after = df_after.groupby('Grade_Category')['Full-Time Equivalent'].sum().reindex(grade_order, fill_value=0)
                
                chart_data_grade = pd.DataFrame({
                    'Grade': grade_order,
                    'Before': grade_before.values,
                    'After': grade_after.values
                })
                
                st.bar_chart(chart_data_grade.set_index('Grade'), color=['#210048', '#005070'], stack=False, height=500)
            
            with col_chart2:
                st.markdown("#### FTE by Years of Service")
                df_before['Service_Category'] = df_before['Length of service (years)'].apply(categorize_service)
                df_after['Service_Category'] = df_after['Length of service (years)'].apply(categorize_service)
                
                service_order = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']
                service_before = df_before.groupby('Service_Category')['Full-Time Equivalent'].sum().reindex(service_order, fill_value=0)
                service_after = df_after.groupby('Service_Category')['Full-Time Equivalent'].sum().reindex(service_order, fill_value=0)
                
                chart_data_service = pd.DataFrame({
                    'Before': service_before.values,
                    'After': service_after.values
                }, index=pd.CategoricalIndex(service_order, categories=service_order, ordered=True))
                
                st.bar_chart(chart_data_service, color=['#210048', '#005070'], stack=False, height=500)
            
            # Chart 3 & 4: FTE and R&T Costs by Faculty (side by side)
            col_chart3, col_chart4 = st.columns(2)
            
            with col_chart3:
                st.markdown("#### FTE by Faculty")
                faculty_before = df_before.groupby('Faculty')['Full-Time Equivalent'].sum().sort_index()
                faculty_after = df_after.groupby('Faculty')['Full-Time Equivalent'].sum().reindex(faculty_before.index, fill_value=0)
                
                chart_data_faculty = pd.DataFrame({
                    'Faculty': faculty_before.index,
                    'Before': faculty_before.values,
                    'After': faculty_after.values
                })
                
                st.bar_chart(chart_data_faculty.set_index('Faculty'), color=['#210048', '#005070'], stack=False, height=500, use_container_width=True)
            
            with col_chart4:
                st.markdown("#### R&T Contract Costs by Faculty (£) ⚠️")
                
                # Calculate R&T costs by faculty
                df_before['RT_Cost'] = df_before.apply(lambda row: get_rt_cost(row['Grade Name']) * row['Full-Time Equivalent'], axis=1)
                df_after['RT_Cost'] = df_after.apply(lambda row: get_rt_cost(row['Grade Name']) * row['Full-Time Equivalent'], axis=1)
                
                cost_before = df_before.groupby('Faculty')['RT_Cost'].sum().sort_index()
                cost_after = df_after.groupby('Faculty')['RT_Cost'].sum().reindex(cost_before.index, fill_value=0)
                
                chart_data_cost = pd.DataFrame({
                    'Faculty': cost_before.index,
                    'Before': cost_before.values,
                    'After': cost_after.values
                })
                
                st.bar_chart(chart_data_cost.set_index('Faculty'), color=['#210048', '#005070'], stack=False, height=500, use_container_width=True)
            
                st.caption("⚠️ R&T Contract Costs are indicative and include mid-point costs and savings for R&T contracts only.\n\n")
        

        
        # Default view with tabs
        else:
            if st.session_state.show_research_groups:
                st.markdown("### Research Groups")
                
                col_analysis, col_back = st.columns([2, 4])
                with col_analysis:
                    if st.button("📊 View Analysis & Charts", key="view_rg_analysis", type="primary", use_container_width=True):
                        st.session_state.show_rg_analysis = True
                        st.session_state.show_research_groups = False
                        st.rerun()
                
                with col_back:
                    if st.button("← Back to Main View", key="back_from_rg"):
                        st.session_state.show_research_groups = False
                        st.rerun()
                
                st.markdown("---")
                
                col_filter_fac, col_filter_sch, col_sort, col_order = st.columns([2, 2, 2, 2])
                
                with col_filter_fac:
                    all_faculties = ['All Faculties'] + sorted(df['Faculty'].dropna().unique().tolist())
                    selected_faculty = st.selectbox(
                        "Filter by Faculty:",
                        all_faculties,
                        key="rg_filter_faculty"
                    )
                
                with col_filter_sch:
                    if selected_faculty == 'All Faculties':
                        available_schools = sorted(df['School'].dropna().unique().tolist())
                    else:
                        available_schools = sorted(df[df['Faculty'] == selected_faculty]['School'].dropna().unique().tolist())
                    
                    all_schools = ['All Schools'] + available_schools
                    selected_school = st.selectbox(
                        "Filter by School:",
                        all_schools,
                        key="rg_filter_school"
                    )
                
                with col_sort:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Name (A-Z)", "Faculty/School", "CoI Total", "Group Size (# Staff)"],
                        key="rg_sort"
                    )
                
                with col_order:
                    sort_order = st.selectbox(
                        "Order:",
                        ["Ascending", "Descending"],
                        key="rg_order"
                    )
                
                all_rgs = get_entities(df, 'Research Group')
                
                rg_data = []
                for rg in all_rgs:
                    rg_staff = get_staff_for_entity(df, 'Research Group', rg)
                    
                    faculty_counts = rg_staff['Faculty'].value_counts()
                    school_counts = rg_staff['School'].value_counts()
                    
                    primary_faculty = faculty_counts.index[0] if len(faculty_counts) > 0 else 'Unknown'
                    primary_school = school_counts.index[0] if len(school_counts) > 0 else 'Unknown'
                    
                    total_coi = rg_staff['CoI income (£)'].sum()
                    staff_count = len(rg_staff)
                    
                    rg_data.append({
                        'name': rg,
                        'faculty': primary_faculty,
                        'school': primary_school,
                        'coi': total_coi,
                        'size': staff_count
                    })
                
                filtered_rg_data = rg_data
                if selected_faculty != 'All Faculties':
                    filtered_rg_data = [rg for rg in filtered_rg_data if rg['faculty'] == selected_faculty]
                
                if selected_school != 'All Schools':
                    filtered_rg_data = [rg for rg in filtered_rg_data if rg['school'] == selected_school]
                
                if sort_option == "Name (A-Z)":
                    filtered_rg_data.sort(key=lambda x: x['name'], reverse=(sort_order == "Descending"))
                elif sort_option == "Faculty/School":
                    filtered_rg_data.sort(key=lambda x: (x['faculty'], x['school'], x['name']), 
                                         reverse=(sort_order == "Descending"))
                elif sort_option == "CoI Total":
                    filtered_rg_data.sort(key=lambda x: x['coi'], reverse=(sort_order == "Descending"))
                else:  # Group Size
                    filtered_rg_data.sort(key=lambda x: x['size'], reverse=(sort_order == "Descending"))
                
                st.markdown(f"**Showing {len(filtered_rg_data)} of {len(rg_data)} research groups**")
                st.markdown("---")
                
                num_cols = 4
                
                if sort_option == "Faculty/School":
                    current_faculty = None
                    current_school = None
                    
                    col_index_tracker = 0 
                    
                    for idx, rg_item in enumerate(filtered_rg_data):
                        
                        if rg_item['faculty'] != current_faculty:
                            current_faculty = rg_item['faculty']
                            current_school = None 
                            st.markdown(f"### 🏛️ {current_faculty}")
                            col_index_tracker = 0
                        
                        if rg_item['school'] != current_school:
                            current_school = rg_item['school']
                            st.markdown(f"#### 🏫 {current_school}")
                            col_index_tracker = 0
                        
                        rg = rg_item['name']
                        key = f"Research Group::{rg}"
                        is_locked = key in st.session_state.locked_entities
                        icon = "🔒" if is_locked else "🟢"
                        
                        col_position = col_index_tracker % num_cols
                        if col_position == 0:
                            cols = st.columns(num_cols)
                        
                        with cols[col_position]:
                            if st.button(f"{icon} {rg}", key=f"rg_{rg}", use_container_width=True):
                                st.session_state.selected_entity = rg
                                st.session_state.selected_level = 'Research Group'
                                # When selecting, load the applied rules into current_criteria for editing
                                st.session_state.current_criteria = deepcopy(st.session_state.locked_entities.get(key, {}).get('criteria', []))
                                st.session_state.show_research_groups = False
                                st.rerun()
                                
                        col_index_tracker += 1
                else:
                    num_groups = len(filtered_rg_data)
                    
                    for row_start in range(0, num_groups, num_cols):
                        cols = st.columns(num_cols)
                        for col_idx, idx in enumerate(range(row_start, min(row_start + num_cols, num_groups))):
                            with cols[col_idx]:
                                rg_item = filtered_rg_data[idx]
                                rg = rg_item['name']
                                key = f"Research Group::{rg}"
                                is_locked = key in st.session_state.locked_entities
                                icon = "🔒" if is_locked else "🟢"
                                
                                label = f"{icon} {rg}"
                                if selected_faculty == 'All Faculties' or selected_school == 'All Schools':
                                    label = f"{icon} {rg}\n({rg_item['school']})"
                                
                                if st.button(label, key=f"rg_{rg}", use_container_width=True):
                                    st.session_state.selected_entity = rg
                                    st.session_state.selected_level = 'Research Group'
                                    st.session_state.current_criteria = deepcopy(st.session_state.locked_entities.get(key, {}).get('criteria', []))
                                    st.session_state.show_research_groups = False
                                    st.rerun()
            elif st.session_state.get('show_rg_analysis', False):
                st.markdown("### Research Groups Analysis")
                
                col_back, _ = st.columns([1, 5])
                with col_back:
                    if st.button("← Back", key="back_from_rg_analysis"):
                        st.session_state.show_rg_analysis = False
                        st.rerun()
                
                st.markdown("---")
                
                rg_analysis = []
                all_rgs = get_entities(df, 'Research Group')
                
                for rg in sorted(all_rgs):
                    rg_staff = get_staff_for_entity(df, 'Research Group', rg)
                    
                    before_count = len(rg_staff)
                    before_fte = rg_staff['Full-Time Equivalent'].sum()
                    
                    rg_excluded_ids = locked_excluded_global.intersection(rg_staff.index)
                    
                    after_count = before_count - len(rg_excluded_ids)
                    after_fte = rg_staff[~rg_staff.index.isin(rg_excluded_ids)]['Full-Time Equivalent'].sum()
                    
                    rg_analysis.append({
                        'Research Group': rg,
                        'Staff Before': before_count,
                        'Staff After': after_count,
                        'Staff Change': after_count - before_count,
                        'FTE Before': f"{before_fte:.2f}",
                        'FTE After': f"{after_fte:.2f}",
                        'FTE Change': f"{after_fte - before_fte:.2f}"
                    })
                
                rg_df = pd.DataFrame(rg_analysis)
                
                st.markdown("#### Research Groups Summary Table")
                st.dataframe(rg_df, use_container_width=True, hide_index=True, height=400)
                
                st.markdown("---")

                st.markdown("#### Staff Count by Research Group")
                
                chart_data = rg_df.sort_values('Staff Before', ascending=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=chart_data['Research Group'],
                    x=chart_data['Staff Before'],
                    name='Before',
                    orientation='h',
                    marker=dict(color='#009BBD'), 
                    text=chart_data['Staff Before'],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    y=chart_data['Research Group'],
                    x=chart_data['Staff After'],
                    name='After',
                    orientation='h',
                    marker=dict(color='#005597'), 
                    text=chart_data['Staff After'],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    barmode='overlay', 
                    height=max(400, len(chart_data) * 30),
                    xaxis_title="Staff Count",
                    yaxis_title="Research Group",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#c4c2c2')
                )
                
                st.plotly_chart(fig, use_container_width=True)
               
            else:
                tab1, tab2 = st.tabs(["Unit Selection", "ALL Staff"])
            
                with tab1:
                    col_fac, col_sch, col_dep, col_rg = st.columns(4)
                    
                    def render_unit_buttons(col, level, heading):
                        with col:
                            st.markdown(f"### {heading}")
                            entities = get_entities(df, level)
                            for entity in entities:
                                key = f"{level}::{entity}"
                                is_locked = key in st.session_state.locked_entities
                                is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
                                
                                icon = "🔒" if is_locked and not is_inherited else "🔓" if is_inherited else "🟢"
                                
                                if level == 'Research Group':
                                    icon = "🔒" if is_locked else "🟢"

                                if st.button(f"{icon} {entity}", key=f"{level.lower()}_{entity}", use_container_width=True):
                                    st.session_state.selected_entity = entity
                                    st.session_state.selected_level = level
                                    # Load applied rules into current_criteria for editing
                                    st.session_state.current_criteria = deepcopy(st.session_state.locked_entities.get(key, {}).get('criteria', []))
                                    st.session_state.show_unlock_warning = False # Not needed anymore, but good practice
                                    st.rerun()
                                    
                    render_unit_buttons(col_fac, 'Faculty', 'Faculties')
                    render_unit_buttons(col_sch, 'School', 'Schools')
                    render_unit_buttons(col_dep, 'Department', 'Departments')
                    
                    with col_rg:
                        st.markdown("### Research Groups")
                        st.markdown("") 
                        st.markdown("") 
                        if st.button("📋 View All Research Groups", key="view_rg_button", use_container_width=True, type="primary"):
                            st.session_state.show_research_groups = True
                            st.rerun()
                            
                with tab2:
                    st.markdown("### Complete Staff List")
                    
                    with st.spinner('Loading staff data...'):
                        
                        display_df = df.copy()
                        display_df['Status'] = display_df.index.map(lambda x: '❌' if x in locked_excluded_global else '✅')
                        
                        exclusion_reason_map = {}
                        for staff_id in df.index:
                            if staff_id in locked_excluded_global:
                                exclusion_reason_map[staff_id] = '; '.join(get_exclusion_reasons(df, staff_id))
                            else:
                                exclusion_reason_map[staff_id] = ''
                                
                        display_df['Exclusion Reason(s)'] = display_df.index.map(exclusion_reason_map)
                        
                        cols_to_show = ['Status', 'Exclusion Reason(s)', 'ID', 'Grade Name', 'Faculty', 
                                    'School', 'Department', 'Full-Time Equivalent', 'Length of service (years)', 'CoI income (£)', 'Scholarly Output', 
                                        'Citations', 'Research Group 1','Research Group 2','Research Group 3','Research Group 4']
                        
                        cols_available = [c for c in cols_to_show if c in display_df.columns]
                    
                    st.dataframe(display_df[cols_available], use_container_width=True, hide_index=True, height=600)

else:
    st.info("📥 Please upload a CSV file to begin analysis")
