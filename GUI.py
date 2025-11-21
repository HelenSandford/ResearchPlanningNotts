import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from copy import deepcopy
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)

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

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'locked_entities' not in st.session_state:
    st.session_state.locked_entities = {}
if 'selected_entity' not in st.session_state:
    st.session_state.selected_entity = None
if 'selected_level' not in st.session_state:
    st.session_state.selected_level = None
if 'preview_criteria' not in st.session_state:
    st.session_state.preview_criteria = []
if 'show_unlock_warning' not in st.session_state:
    st.session_state.show_unlock_warning = False
if 'inherited_from' not in st.session_state:
    st.session_state.inherited_from = None
if 'show_research_groups' not in st.session_state:
    st.session_state.show_research_groups = False
if 'show_rg_analysis' not in st.session_state: 
    st.session_state.show_rg_analysis = False     

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
                    parents.append(('Faculty', faculty, st.session_state.locked_entities[fac_key]))
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
        # Check Faculty lock
        if pd.notna(staff_row.get('Faculty')):
            fac_key = f"Faculty::{staff_row['Faculty']}"
            if fac_key in st.session_state.locked_entities:
                affected_locks.append(('Faculty', staff_row['Faculty']))
        # Check School lock
        if pd.notna(staff_row.get('School')):
            sch_key = f"School::{staff_row['School']}"
            if sch_key in st.session_state.locked_entities:
                affected_locks.append(('School', staff_row['School']))
        # Check Department lock
        if pd.notna(staff_row.get('Department')):
            dept_key = f"Department::{staff_row['Department']}"
            if dept_key in st.session_state.locked_entities:
                affected_locks.append(('Department', staff_row['Department']))
    
    # Return unique locks
    return list(set(affected_locks))

def lock_entity_and_children(level, entity, criteria):
    """Lock an entity - Department locks are NOT automatically created"""
    key = f"{level}::{entity}"
    st.session_state.locked_entities[key] = {
        'criteria': deepcopy(criteria),
        'timestamp': datetime.now().isoformat(),
        'inherited_from': None
    }
    
    # NO automatic child locking - users must explicitly lock Departments if desired

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
        
        # Normalize metrics by FTE for fair comparison
        normalized_df = filtered.copy()
        for metric in sort_metrics:
            if metric in filtered.columns and 'Full-Time Equivalent' in filtered.columns:
                # Create normalized column (metric per FTE)
                normalized_df[f'{metric}_normalized'] = filtered[metric] / filtered['Full-Time Equivalent']
        
        # Sort by normalized metrics
        normalized_cols = [f'{m}_normalized' for m in sort_metrics]
        ascending_list = [True] * len(normalized_cols)
        normalized_df = normalized_df.sort_values(by=normalized_cols, ascending=ascending_list)
        
        # Calculate FTE-based exclusion
        total_fte = normalized_df['Full-Time Equivalent'].sum()
        target_fte_to_exclude = total_fte * percent / 100
        
        # Accumulate staff until we reach the target FTE
        # Stop BEFORE exceeding the target percentage
        cumulative_fte = 0
        to_exclude = []
        for idx, row in normalized_df.iterrows():
            # Check if adding this person would exceed the target
            if cumulative_fte + row['Full-Time Equivalent'] <= target_fte_to_exclude:
                to_exclude.append(idx)
                cumulative_fte += row['Full-Time Equivalent']
            else:
                # Would exceed target, so stop here
                break
        
        return set(to_exclude)
    
    return set()

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
    """Create a standardized metrics comparison table"""
    metrics_data = {
        'Metric': ['Staff Count', 'Total FTE', 'Total CoI (£)', 'Avg CoI/FTE (£)',
                  'Total Scholarly Output', 'Avg Scholarly/FTE', 'Total Citations',
                  'Avg Citations/FTE', 'Citations per Publication', 'Total R&T Cost (£) ⚠️',
                  'Total Active PGRs', 'PGR per FTE', 'Total ESE Contact Hours', 'ESE Contact Hours per FTE'
                  ],
        'Before': [
            before_metrics['count'], f"{before_metrics['fte']:.2f}", f"£{before_metrics['total_coi']:,.0f}",
            f"£{before_metrics['coi_per_fte']:,.0f}", f"{before_metrics['total_schol']:.0f}",
            f"{before_metrics['schol_per_fte']:.2f}", f"{before_metrics['total_cit']:.0f}",
            f"{before_metrics['cit_per_fte']:.0f}", f"{before_metrics['cit_per_pub']:.2f}",
            f"£{before_metrics['total_rt_cost']:,.0f}", f"{before_metrics['total_pgr']:.0f}",
            f"{before_metrics['pgr_per_fte']:.1f}", f"{before_metrics['total_ese_contact']:,.0f}", 
            f"{before_metrics['ese_per_fte']:,.0f}"
        ],
        'After': [
            after_metrics['count'], f"{after_metrics['fte']:.2f}", f"£{after_metrics['total_coi']:,.0f}",
            f"£{after_metrics['coi_per_fte']:,.0f}", f"{after_metrics['total_schol']:.0f}",
            f"{after_metrics['schol_per_fte']:.2f}", f"{after_metrics['total_cit']:.0f}",
            f"{after_metrics['cit_per_fte']:.0f}", f"{after_metrics['cit_per_pub']:.2f}",
            f"£{after_metrics['total_rt_cost']:,.0f}", f"{after_metrics['total_pgr']:.0f}",
            f"{after_metrics['pgr_per_fte']:.1f}", f"{after_metrics['total_ese_contact']:,.0f}", 
            f"{after_metrics['ese_per_fte']:,.0f}"
        ],
        'Change': [
            after_metrics['count'] - before_metrics['count'], 
            f"{after_metrics['fte'] - before_metrics['fte']:.2f}",
            f"£{after_metrics['total_coi'] - before_metrics['total_coi']:,.0f}",
            f"£{after_metrics['coi_per_fte'] - before_metrics['coi_per_fte']:,.0f}",
            f"{after_metrics['total_schol'] - before_metrics['total_schol']:.0f}",
            f"{after_metrics['schol_per_fte'] - before_metrics['schol_per_fte']:.2f}",
            f"{after_metrics['total_cit'] - before_metrics['total_cit']:.0f}",
            f"{after_metrics['cit_per_fte'] - before_metrics['cit_per_fte']:.0f}",
            f"{after_metrics['cit_per_pub'] - before_metrics['cit_per_pub']:.2f}",
            f"£{after_metrics['total_rt_cost'] - before_metrics['total_rt_cost']:,.0f}",
            f"{after_metrics['total_pgr'] - before_metrics['total_pgr']:.0f}",
            f"{after_metrics['pgr_per_fte'] - before_metrics['pgr_per_fte']:.0f}",
            f"{after_metrics['total_ese_contact'] - before_metrics['total_ese_contact']:,.0f}",
            f"{after_metrics['ese_per_fte'] - before_metrics['ese_per_fte']:,.0f}"                        
        ]
    }
    return pd.DataFrame(metrics_data)

def calculate_metrics(staff_data, excluded_ids=None, original_staff_data=None):
    if excluded_ids is None:
        excluded_ids = set()
    included = staff_data[~staff_data.index.isin(excluded_ids)]
    excluded = staff_data[staff_data.index.isin(excluded_ids)]
    
    if len(included) == 0:
        return {'count': 0, 'fte': 0, 'total_coi': 0, 'coi_per_fte': 0,
                'total_schol': 0, 'schol_per_fte': 0, 'total_cit': 0,
                'cit_per_fte': 0, 'cit_per_pub': 0, 'total_rt_cost': 0,
                'total_pgr': 0, 'pgr_per_fte': 0, 'total_ese_contact': 0, 'ese_per_fte': 0}
    
    total_fte = included['Full-Time Equivalent'].sum()
    total_coi = included['CoI income (£)'].sum()
    total_schol = included['Scholarly Output'].sum()
    total_cit = included['Citations'].sum()
    
    # Calculate R&T costs for included staff
    total_rt_cost = sum(get_rt_cost(row['Grade Name']) * row['Full-Time Equivalent'] 
                        for _, row in included.iterrows())
    
    # For PGR and ESE, use original staff data if provided (for "after" calculations)
    # This ensures these totals remain constant
    data_for_pgr_ese = original_staff_data if original_staff_data is not None else staff_data
    
    # Calculate PGR students from original data
    total_pgr = data_for_pgr_ese['PGR Active students'].sum() if 'PGR Active students' in data_for_pgr_ese.columns else 0
    
    # Calculate ESE Contact Hours from original data (Hours * Sessions * Headcount)
    total_ese_contact = 0
    if all(col in data_for_pgr_ese.columns for col in ['ESE Hours Timetabled', 'ESE Sessions Timetabled', 'ESE Headcount Timetabled']):
        data_for_pgr_ese_copy = data_for_pgr_ese.copy()
        data_for_pgr_ese_copy['ESE_Contact_Hours'] = (data_for_pgr_ese_copy['ESE Hours Timetabled'] * 
                                         data_for_pgr_ese_copy['ESE Sessions Timetabled'] * 
                                         data_for_pgr_ese_copy['ESE Headcount Timetabled'])
        total_ese_contact = data_for_pgr_ese_copy['ESE_Contact_Hours'].sum()
    
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
        lock_staff = get_staff_for_entity(df, lock_level, lock_entity)
        lock_excluded = apply_criteria(lock_staff, lock_data['criteria'])
        
        # For all staff in this entity, record their exclusion status
        # Higher priority locks OVERWRITE lower priority ones
        for staff_id in lock_staff.index:
            staff_exclusions[staff_id] = (staff_id in lock_excluded, lock_key)
    
    # Return all staff marked as excluded
    return {staff_id for staff_id, (is_excluded, _) in staff_exclusions.items() if is_excluded}

def get_exclusion_reasons(df, staff_id):
    reasons = []
    for lock_key, lock_data in st.session_state.locked_entities.items():
        lock_level, lock_entity = lock_key.split('::')
        lock_staff = get_staff_for_entity(df, lock_level, lock_entity)
        if staff_id in lock_staff.index:
            lock_excluded = apply_criteria(lock_staff, lock_data['criteria'])
            if staff_id in lock_excluded:
                inherited = lock_data.get('inherited_from')
                if inherited:
                    parent_level, parent_entity = inherited.split('::')
                    reasons.append(f"{lock_entity} ({lock_level}) [inherited from {parent_entity}]")
                else:
                    reasons.append(f"{lock_entity} ({lock_level})")
    return reasons

# Header row
col_logo, col_title, col_upload = st.columns([2, 3, 2])

with col_logo:
    st.markdown("""
        <div style='display: flex; align-items: center;'>
            <img src='data:image/png;base64,{}' width='75' style='margin-right: 15px;'/>
            <span style='color: #E8E8E8; font-size: 16px; font-weight: 600; letter-spacing: 0.5px;'>
                SEA Consultancy Ltd<span style='font-size: 12px; font-weight: 400;'> © 2025</span>
            </span>
        </div>
    """.format(__import__('base64').b64encode(open("logo.png", "rb").read()).decode()), 
    unsafe_allow_html=True)

with col_title:
    st.title("Research Planning Dashboard")

with col_upload:
    if st.session_state.data is None:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                numeric_cols = ['Full-Time Equivalent', 'Length of service (years)', 'CoI income (£)',
                              'Nr of research projects', 'Scholarly Output', 'Citations',
                              'Citations per Publication']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                st.session_state.data = df
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file")
    else:
        st.markdown(f"<div class='uploaded-file-info'>📊 {len(st.session_state.data)} records loaded</div>", unsafe_allow_html=True)
        if st.button("📤 Upload Different File", use_container_width=True):
            st.session_state.data = None
            st.session_state.locked_entities = {}
            st.session_state.selected_entity = None
            st.session_state.selected_level = None
            st.session_state.preview_criteria = []
            st.rerun()

st.markdown("---")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Calculate overall metrics
    locked_excluded = get_locked_excluded_ids(df)
    before = calculate_metrics(df)
    after = calculate_metrics(df, locked_excluded)
    
    # Main layout: Summary panel on left, content on right
    col_summary, col_main = st.columns([1, 4])
    
    # LEFT: Summary Panel
    with col_summary:
        if st.session_state.selected_entity and st.session_state.selected_level:
            st.markdown(f"### {st.session_state.selected_entity} Summary")
            entity_staff = get_staff_for_entity(df, st.session_state.selected_level, st.session_state.selected_entity)
            preview_excluded = apply_criteria(entity_staff, st.session_state.preview_criteria)
            entity_before = calculate_metrics(entity_staff)
            entity_after = calculate_metrics(entity_staff, preview_excluded)
            
            # Calculate deltas
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
                st.session_state.preview_criteria = []
                st.session_state.show_unlock_warning = False
                st.session_state.inherited_from = None
                st.rerun()
        else:
            st.markdown("### Overall Summary")
            
            # Calculate deltas
            staff_delta = after['count'] - before['count']
            fte_delta = after['fte'] - before['fte']
            coi_delta = after['coi_per_fte'] - before['coi_per_fte']
            schol_delta = after['schol_per_fte'] - before['schol_per_fte']
            
            st.metric("Staff", after['count'],
                     delta=f"{staff_delta:+d}" if staff_delta != 0 else None,
                     delta_color="inverse")
            st.metric("FTE", f"{after['fte']:.1f}",
                     delta=f"{fte_delta:+.1f}" if fte_delta != 0 else None,
                     delta_color="inverse")
            st.metric("CoI/FTE", f"£{after['coi_per_fte']/1000:.0f}k",
                     delta=f"{coi_delta/1000:+.0f}k" if coi_delta != 0 else None,
                     delta_color="normal")
            st.metric("ScOutput/FTE", f"{after['schol_per_fte']:.2f}",
                     delta=f"{schol_delta:+.2f}" if schol_delta != 0 else None,
                     delta_color="normal")
            st.metric("Locked", len(st.session_state.locked_entities))
            
            st.markdown("---")
            if st.button("Detailed Report", use_container_width=True, type="primary"):
                st.session_state.show_detailed_report = not st.session_state.get('show_detailed_report', False)
                st.rerun()
    
    # RIGHT: Main Content
    with col_main:
        # Show Unit Modification View if unit selected
        if st.session_state.selected_entity and st.session_state.selected_level:
            entity = st.session_state.selected_entity
            level = st.session_state.selected_level
            key = f"{level}::{entity}"
            is_locked = key in st.session_state.locked_entities
            
            # Check for parent locks (different for Research Groups)
            if level == 'Research Group':
                parent_locks_info = get_parent_locks_for_research_group(df, entity)
                has_parent_lock = len(parent_locks_info) > 0
                is_inherited = False  # Research groups don't inherit, they override
            else:
                parent_locks = get_parent_locks(level, entity)
                has_parent_lock = len(parent_locks) > 0
                is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
            
            st.markdown(f"# {entity}")
            st.caption(f"{level}")
            
            # Show parent lock information for Departments
            if level == 'Department' and has_parent_lock and not st.session_state.show_unlock_warning:
                parent_level, parent_entity, parent_data = parent_locks[0]
                
                # Count how many staff would be affected by override
                dept_staff = get_staff_for_entity(df, 'Department', entity)
                school_excluded = apply_criteria(dept_staff, parent_data['criteria'])
                
                st.markdown(f"""
                <div class='warning-box'>
                    <strong>⚠️ Override Parent School Rules?</strong><br>
                    This Department is part of <strong>{parent_entity}</strong> which has exclusion rules applied.<br>
                    <small>• {len(school_excluded)} staff in this Department are currently excluded by {parent_entity} rules</small><br>
                    <small>• Applying new Department rules will override the {parent_entity} rules for ALL {len(dept_staff)} staff in this Department</small>
                </div>
                """, unsafe_allow_html=True)

            # Show parent lock information for Research Groups
            elif level == 'Research Group' and has_parent_lock and not st.session_state.show_unlock_warning:
                unique_locks = list(set(parent_locks_info))
                lock_text = ', '.join([f"{entity} ({lvl})" for lvl, entity in unique_locks])
                st.markdown(f"""
                <div class='warning-box'>
                    <strong>⚠️ Warning:</strong> Some staff in this research group have existing rules from: {lock_text}<br>
                    <small>Rules applied to this research group will override previous Faculty/School/Department rules for these staff members.</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Show parent lock information for other levels
            elif level != 'Research Group' and has_parent_lock and not st.session_state.show_unlock_warning:
                for parent_level, parent_entity, parent_data in parent_locks:
                    st.markdown(f"""
                    <div class='inherited-rules'>
                        <strong>ℹ️ Rules Inherited From:</strong> {parent_entity} ({parent_level})<br>
                        <small>This {level.lower()} has rules applied from a higher level.</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            entity_staff = get_staff_for_entity(df, level, entity)
            preview_excluded = apply_criteria(entity_staff, st.session_state.preview_criteria)
            entity_before = calculate_metrics(entity_staff)
            entity_after = calculate_metrics(entity_staff, preview_excluded)
            
            st.markdown("---")
            
            # Handle unlock warning dialog
            if st.session_state.show_unlock_warning and has_parent_lock and level != 'Research Group':
                st.markdown("""
                <div class='warning-box'>
                    <h3>⚠️ Override Parent Rules?</h3>
                    <p>This will override rules applied at a higher level. You can modify or remove the inherited rules for this specific unit.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show inherited rules
                st.markdown("### Inherited Rules:")
                parent_level, parent_entity, parent_data = parent_locks[0]
                for idx, criterion in enumerate(parent_data['criteria']):
                    with st.expander(f"Rule {idx + 1} from {parent_entity}"):
                        if criterion.get('grades'):
                            st.write(f"**Grades:** {', '.join(criterion['grades'])}")
                        if criterion.get('service_years'):
                            op, val = criterion['service_years']
                            st.write(f"**Service Years:** {op} {val}")
                        if criterion.get('bottom_percentile'):
                            st.write(f"**Bottom Percentile:** {criterion['bottom_percentile']}%")
                        if criterion.get('sort_by'):
                            st.write(f"**Sort By:** {' → '.join(criterion['sort_by'])}")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✓ Continue & Override", use_container_width=True, type="primary"):
                        # Copy parent criteria to preview for editing
                        parent_level, parent_entity, parent_data = parent_locks[0]
                        st.session_state.preview_criteria = deepcopy(parent_data['criteria'])
                        st.session_state.show_unlock_warning = False
                        # Remove the inherited lock completely so it can be modified
                        unlock_entity_only(level, entity)
                        # Also remove any child inherited locks
                        children = get_children(level, entity)
                        for child_level, child_entity in children:
                            unlock_entity_only(child_level, child_entity)
                        st.rerun()                
                with col_cancel:
                    if st.button("✗ Cancel", use_container_width=True):
                        st.session_state.show_unlock_warning = False
                        st.session_state.selected_entity = None
                        st.session_state.selected_level = None
                        st.rerun()
            
            else:
                # Normal rule management interface
                col_action1, col_action2, _ = st.columns([1, 1, 3])
                with col_action1:
                    if not is_locked and st.button("+ Add Rule", use_container_width=True):
                        st.session_state.preview_criteria.append({})
                        st.rerun()
                with col_action2:
                    if not is_locked:
                        if st.button("Lock All & Apply", use_container_width=True, type="primary",
                                   disabled=len(st.session_state.preview_criteria) == 0):
                            if level == 'Research Group':
                                # Research groups don't cascade, they just lock themselves
                                st.session_state.locked_entities[key] = {
                                    'criteria': deepcopy(st.session_state.preview_criteria),
                                    'timestamp': datetime.now().isoformat(),
                                    'inherited_from': None
                                }
                                st.success(f"Locked: {entity}")
                            else:
                                lock_entity_and_children(level, entity, st.session_state.preview_criteria)
                                st.success(f"Locked: {entity} and all sub-units")
                            st.rerun()
                    else:
                        if is_inherited and not st.session_state.show_unlock_warning and level != 'Research Group':
                            if st.button("Unlock & Override Parent", use_container_width=True):
                                st.session_state.show_unlock_warning = True
                                st.rerun()
                        elif not is_inherited or level == 'Research Group':
                            if st.button("Unlock to Modify", use_container_width=True):
                                st.session_state.preview_criteria = deepcopy(st.session_state.locked_entities[key]['criteria'])
                                unlock_entity_only(level, entity)
                                st.success(f"Unlocked: {entity}")
                                st.rerun()
                
                st.markdown("### Exclusion Rules")
                
                # Display rules in columns (max 3 per row)
                if st.session_state.preview_criteria:
                    num_rules = len(st.session_state.preview_criteria)
                    rules_per_row = 3
                    
                    for row_start in range(0, num_rules, rules_per_row):
                        cols = st.columns(rules_per_row)
                        for col_idx, idx in enumerate(range(row_start, min(row_start + rules_per_row, num_rules))):
                            with cols[col_idx]:
                                criterion = st.session_state.preview_criteria[idx]
                                
                                st.markdown(f"**Rule {idx + 1}**")
                                
                                if not is_locked:
                                    if st.button("Delete", key=f"del_{idx}"):
                                        st.session_state.preview_criteria.pop(idx)
                                        st.rerun()
                                
                                grade_options = ['RT5*', 'RT6*', 'RT7*', 'Clinical', 
                                               'RT5-A', 'RT5E-A', 'RT6-A', 'RT6-R', 'RT7-A', 'RT7-R']
                                selected_grades = st.multiselect("Grades", grade_options,
                                    default=criterion.get('grades', []),
                                    key=f"grades_{idx}", disabled=is_locked,
                                    label_visibility="collapsed",
                                    placeholder="Choose grades or leave blank to select ALL")
                                if selected_grades != criterion.get('grades', []):
                                    st.session_state.preview_criteria[idx]['grades'] = selected_grades
                                    st.rerun()
                                
                                st.caption("Years of Service")
                                col_op, col_val = st.columns([1, 2])
                                with col_op:
                                    service_op = st.selectbox("Operator", ['', '>', '<'],
                                        index=['', '>', '<'].index(criterion.get('service_years', ('', 0))[0]) if criterion.get('service_years') else 0,
                                        key=f"service_op_{idx}", disabled=is_locked,
                                        label_visibility="collapsed")
                                with col_val:
                                    service_val = st.number_input("Years", min_value=0.0, step=1.0,
                                        value=float(criterion.get('service_years', ('', 0))[1]),
                                        key=f"service_val_{idx}", disabled=is_locked,
                                        label_visibility="collapsed")
                                
                                if service_op and (service_op, service_val) != criterion.get('service_years'):
                                    st.session_state.preview_criteria[idx]['service_years'] = (service_op, service_val)
                                    st.rerun()
                                
                                st.caption("Sort by %")
                                percent = st.number_input("Bottom %", min_value=0, max_value=100,
                                    value=criterion.get('bottom_percentile', 0),
                                    key=f"percent_{idx}", disabled=is_locked,
                                    label_visibility="collapsed")
                                if percent != criterion.get('bottom_percentile', 0):
                                    st.session_state.preview_criteria[idx]['bottom_percentile'] = percent
                                    st.rerun()
                                
                                metric_options = ['CoI income (£)', 'Scholarly Output', 'Citations']
                                selected_metrics = st.multiselect("Metrics", metric_options,
                                    default=criterion.get('sort_by', []),
                                    key=f"metrics_{idx}", disabled=is_locked,
                                    label_visibility="collapsed",
                                    placeholder="Exclude bottom % by... (multiselect in preference order)")
                                
                                if selected_metrics != criterion.get('sort_by', []):
                                    st.session_state.preview_criteria[idx]['sort_by'] = selected_metrics
                                    st.rerun()
                                
                                if selected_metrics:
                                    st.caption(" → ".join([f"{i+1}. {m.split('(')[0].strip()}" for i, m in enumerate(selected_metrics)]))
                else:
                    st.info("No rules defined. Click '+ Add Rule' to create one.")
                
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

            metrics_table = create_metrics_table(before, after)
            st.dataframe(metrics_table, use_container_width=True, hide_index=True,  height=len(metrics_table) * 35 + 38)
            
            st.markdown("---")
            st.markdown("### Visual Analysis")
            
            # Helper function to categorize grades
            def categorize_grade(grade):
                if pd.isna(grade):
                    return 'Other'
                grade_str = str(grade)
                if grade_str.startswith('RT5'):
                    return 'RT5'
                elif grade_str.startswith('RT6'):
                    return 'RT6'
                elif grade_str.startswith('RT7'):
                    return 'RT7'
                elif grade_str.startswith('CL'):
                    return 'Clinical'
                else:
                    return 'Other'
            
            # Helper function to categorize service years
            def categorize_service(years):
                if years < 5:
                    return '0-5'
                elif years < 10:
                    return '5-10'
                elif years < 15:
                    return '10-15'
                elif years < 20:
                    return '15-20'
                elif years < 25:
                    return '20-25'
                elif years < 30:
                    return '25-30'
                else:
                    return '30+'
            
            # Prepare data
            df_before = df.copy()
            df_after = df[~df.index.isin(locked_excluded)].copy()
            
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
            # Check if showing research groups grid
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
                
                col_sort, col_order = st.columns([2, 2])
                
                with col_sort:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Name (A-Z)", "CoI Total", "Group Size (# Staff)"],
                        key="rg_sort"
                    )
                
                with col_order:
                    sort_order = st.selectbox(
                        "Order:",
                        ["Ascending", "Descending"],
                        key="rg_order"
                    )
                
                # Get all research groups first
                all_rgs = get_entities(df, 'Research Group')
                
                # Build list with metrics for sorting
                rg_data = []
                for rg in all_rgs:
                    rg_staff = get_staff_for_entity(df, 'Research Group', rg)
                    total_coi = rg_staff['CoI income (£)'].sum()
                    staff_count = len(rg_staff)
                    rg_data.append({
                        'name': rg,
                        'coi': total_coi,
                        'size': staff_count
                    })
                
                # Sort based on selection
                if sort_option == "Name (A-Z)":
                    rg_data.sort(key=lambda x: x['name'], reverse=(sort_order == "Descending"))
                elif sort_option == "CoI Total":
                    rg_data.sort(key=lambda x: x['coi'], reverse=(sort_order == "Descending"))
                else:  # Group Size
                    rg_data.sort(key=lambda x: x['size'], reverse=(sort_order == "Descending"))
                
                research_groups = [item['name'] for item in rg_data]
                
                # Display in grid format (4 columns)
                num_cols = 4
                num_groups = len(research_groups)
                
                for row_start in range(0, num_groups, num_cols):
                    cols = st.columns(num_cols)
                    for col_idx, idx in enumerate(range(row_start, min(row_start + num_cols, num_groups))):
                        with cols[col_idx]:
                            rg = research_groups[idx]
                            key = f"Research Group::{rg}"
                            is_locked = key in st.session_state.locked_entities
                            icon = "🔒" if is_locked else "🟢"
                            if st.button(f"{icon} {rg}", key=f"rg_{rg}", use_container_width=True):
                                st.session_state.selected_entity = rg
                                st.session_state.selected_level = 'Research Group'
                                if is_locked:
                                    st.session_state.preview_criteria = deepcopy(st.session_state.locked_entities[key]['criteria'])
                                else:
                                    st.session_state.preview_criteria = []
                                st.session_state.show_research_groups = False
                                st.rerun()
                                
                        # Show Research Groups Analysis if button clicked
            elif st.session_state.get('show_rg_analysis', False):
                st.markdown("### Research Groups Analysis")
                
                col_back, _ = st.columns([1, 5])
                with col_back:
                    if st.button("← Back", key="back_from_rg_analysis"):
                        st.session_state.show_rg_analysis = False
                        st.rerun()
                
                st.markdown("---")
                
                # Build research group analysis data
                rg_analysis = []
                all_rgs = get_entities(df, 'Research Group')
                
                for rg in sorted(all_rgs):
                    rg_staff = get_staff_for_entity(df, 'Research Group', rg)
                    
                    # Calculate before metrics (all staff in group)
                    before_count = len(rg_staff)
                    before_fte = rg_staff['Full-Time Equivalent'].sum()
                    
                    # Calculate after metrics (excluding locked exclusions)
                    rg_excluded = set()
                    key = f"Research Group::{rg}"
                    if key in st.session_state.locked_entities:
                        # Direct RG lock
                        rg_excluded = apply_criteria(rg_staff, st.session_state.locked_entities[key]['criteria'])
                    else:
                        # Check for Faculty/School/Department locks affecting this RG's staff
                        for staff_id in rg_staff.index:
                            if staff_id in locked_excluded:
                                rg_excluded.add(staff_id)
                    
                    after_count = before_count - len(rg_excluded)
                    after_fte = rg_staff[~rg_staff.index.isin(rg_excluded)]['Full-Time Equivalent'].sum()
                    
                    rg_analysis.append({
                        'Research Group': rg,
                        'Staff Before': before_count,
                        'Staff After': after_count,
                        'Staff Change': after_count - before_count,
                        'FTE Before': f"{before_fte:.2f}",
                        'FTE After': f"{after_fte:.2f}",
                        'FTE Change': f"{after_fte - before_fte:.2f}"
                    })
                
                # Create DataFrame
                rg_df = pd.DataFrame(rg_analysis)
                
                # Display table
                st.markdown("#### Research Groups Summary Table")
                st.dataframe(rg_df, use_container_width=True, hide_index=True, height=400)
                
                st.markdown("---")

                # Create chart data
                st.markdown("#### Staff Count by Research Group")
                
                # Prepare data sorted by Before count
                chart_data = rg_df.sort_values('Staff Before', ascending=True)
                
                # Create overlapping horizontal bar chart using Plotly
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add Before bars (light blue, full width)
                fig.add_trace(go.Bar(
                    y=chart_data['Research Group'],
                    x=chart_data['Staff Before'],
                    name='Before',
                    orientation='h',
                    marker=dict(color='#87CEEB'),
                    text=chart_data['Staff Before'],
                    textposition='outside'
                ))
                
                # Add After bars (darker blue, overlay on top)
                fig.add_trace(go.Bar(
                    y=chart_data['Research Group'],
                    x=chart_data['Staff After'],
                    name='After',
                    orientation='h',
                    marker=dict(color='#4682B4'),
                    text=chart_data['Staff After'],
                    textposition='outside'
                ))
                
                # Update layout for overlay effect
                fig.update_layout(
                    barmode='overlay',  # This creates the overlay effect
                    height=max(400, len(chart_data) * 30),
                    xaxis_title="Staff Count",
                    yaxis_title="Research Group",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e8e8e8')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("💡 Light blue shows original staff count, darker blue overlay shows remaining staff after exclusions")
               
            else:
                tab1, tab2 = st.tabs(["Unit Selection", "ALL Staff"])
            
                # TAB 1: Unit Selection
                with tab1:
                    col_fac, col_sch, col_dep, col_rg = st.columns(4)
                    
                    with col_fac:
                        st.markdown("### Faculties")
                        faculties = get_entities(df, 'Faculty')
                        for fac in faculties:
                            key = f"Faculty::{fac}"
                            is_locked = key in st.session_state.locked_entities
                            is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
                            icon = "🔒" if is_locked and not is_inherited else "🔓" if is_inherited else "🟢"
                            if st.button(f"{icon} {fac}", key=f"fac_{fac}", use_container_width=True):
                                st.session_state.selected_entity = fac
                                st.session_state.selected_level = 'Faculty'
                                if is_locked:
                                    st.session_state.preview_criteria = deepcopy(st.session_state.locked_entities[key]['criteria'])
                                else:
                                    st.session_state.preview_criteria = []
                                st.rerun()
                    
                    with col_sch:
                        st.markdown("### Schools")
                        schools = get_entities(df, 'School')
                        for sch in schools:
                            key = f"School::{sch}"
                            is_locked = key in st.session_state.locked_entities
                            is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
                            icon = "🔒" if is_locked and not is_inherited else "🔓" if is_inherited else "🟢"
                            if st.button(f"{icon} {sch}", key=f"sch_{sch}", use_container_width=True):
                                st.session_state.selected_entity = sch
                                st.session_state.selected_level = 'School'
                                if is_locked:
                                    st.session_state.preview_criteria = deepcopy(st.session_state.locked_entities[key]['criteria'])
                                else:
                                    st.session_state.preview_criteria = []
                                st.rerun()
                    
                    with col_dep:
                        st.markdown("### Departments")
                        departments = get_entities(df, 'Department')
                        for dep in departments:
                            key = f"Department::{dep}"
                            is_locked = key in st.session_state.locked_entities
                            is_inherited = is_locked and st.session_state.locked_entities[key].get('inherited_from') is not None
                            icon = "🔒" if is_locked and not is_inherited else "🔓" if is_inherited else "🟢"
                            if st.button(f"{icon} {dep}", key=f"dep_{dep}", use_container_width=True):
                                st.session_state.selected_entity = dep
                                st.session_state.selected_level = 'Department'
                                if is_locked:
                                    st.session_state.preview_criteria = deepcopy(st.session_state.locked_entities[key]['criteria'])
                                else:
                                    st.session_state.preview_criteria = []
                                st.rerun()
                    
                    with col_rg:
                        st.markdown("### Research Groups")
                        st.markdown("")  # Spacing
                        st.markdown("")  # Spacing
                        if st.button("📋 View All Research Groups", key="view_rg_button", use_container_width=True, type="primary"):
                            st.session_state.show_research_groups = True
                            st.rerun()
                # TAB 2: ALL Staff
                with tab2:
                    st.markdown("### Complete Staff List")
                    
                    with st.spinner('Loading staff data...'):
                        # Optimize: Pre-calculate all exclusion reasons once
                        exclusion_map = {}
                        for staff_id in df.index:
                            if staff_id in locked_excluded:
                                exclusion_map[staff_id] = '; '.join(get_exclusion_reasons(df, staff_id))
                            else:
                                exclusion_map[staff_id] = ''
                        
                        display_df = df.copy()
                        display_df['Status'] = display_df.index.map(lambda x: '❌' if x in locked_excluded else '✅')
                        display_df['Exclusion Reason(s)'] = display_df.index.map(exclusion_map)
                        
                        cols_to_show = ['Status', 'Exclusion Reason(s)', 'ID', 'Grade Name', 'Faculty', 
                                    'School', 'Department', 'Full-Time Equivalent', 'Length of service (years)', 'CoI income (£)', 'Scholarly Output', 
                                        'Citations', 'Research Group 1','Research Group 2','Research Group 3','Research Group 4']
                        
                        cols_available = [c for c in cols_to_show if c in display_df.columns]
                    
                    st.dataframe(display_df[cols_available], use_container_width=True, hide_index=True, height=600)

else:
    st.info("📥 Please upload a CSV file to begin analysis")
