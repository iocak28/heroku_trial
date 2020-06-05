import pytest
import plotly
import os
import sys
sys.path.append(os.getcwd() + '/..')

@pytest.fixture(autouse=True)
def teardown():
    d = os.path.dirname(os.path.abspath('test_util.py'))
    d = d.split('/')
    if d[-1]!='test':
       os.chdir('test')

def test_dashboard():
    '''
    Tests dashboard.py.
    '''
    os.chdir('..')
    import dashboard
    try:
        assert type(dashboard.marital_state_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.marital_status_probab()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.education_level_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.education_level_prob()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.income_level_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.job_prob()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.contact_way_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.contact_prob()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.loan_status()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.loan_prob()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.house_status_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.house_prob()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.prediction_pie_chart()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.predicted_prob_hist()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.age_distribution()) == plotly.graph_objs._figure.Figure
        assert type(dashboard.age_prob()) == plotly.graph_objs._figure.Figure
        
        
    except:
        raise("Error in Dashboard...!!!")
