import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from textwrap import dedent as d
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import plot
import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign
#from sklearn.externals import joblib
import joblib
from util import dynamic_predict

from visualization import analysis  
# from visualization import plots 

csv_path = 'data/bank-additional-full.csv'
my_analysis = analysis.Analysis(csv_path)
myList, labels = my_analysis.map_age()

# Read and modify prediction data
predictions = pd.read_csv('data/predictions.csv')

df = predictions.copy()

df = df[['customer_id', 'age', 'job_transformed', 'poutcome', 'pred', 'prob_1']]  # prune columns for example
df.sort_values(by = ['prob_1'], ascending = False, inplace = True)
df['prob_1'] = np.around(df['prob_1'], decimals = 2)
df['is_called'] = 'Not Called'

df.loc[df['job_transformed'] == 'no_income', 'job_transformed'] = 'No Income'
df.loc[df['job_transformed'] == 'higher_income', 'job_transformed'] = 'Higher Income'
df.loc[df['job_transformed'] == 'lower_income', 'job_transformed'] = 'Lower Income'
df.loc[df['job_transformed'] == 'unknown', 'job_transformed'] = 'Unknown'

df.loc[df['poutcome'] == 'success', 'poutcome'] = 'Success'
df.loc[df['poutcome'] == 'nonexistent', 'poutcome'] = 'None'
df.loc[df['poutcome'] == 'failure', 'poutcome'] = 'Failure'

def marital_state_distribution():
    '''
    This function gives the plot of distribution of people based on marital status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.


    '''
    percents = my_analysis.percentage_of_population('marital')
    v = my_analysis.get_count('marital')['y']
    values = [v[1], v[0], v[2], v[3]]
    labels = ['Married', 'Divorced', 'Single', 'Unknown']
    my_analysis.get_count('marital')
    explode = (0.2, 0, 0)
    fig = px.pie(percents,  values= values, names = labels, 
                title = '% of Population based on marital status')
    return fig
 
def marital_status_probab():
    '''
    This function gives the plot of probability of success based on people's marital status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    marital_status_probab = my_analysis.get_probabilities('marital')
    data = marital_status_probab
    data['y'] = data['y']*100
    fig = px.bar(data, x='marital', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'marital': 'Marital Status'},
                height=400, title = 'Probability of success by marital status')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def education_level_distribution():
    '''
    This function gives plot of distribution of people based on education level.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    
    percents = my_analysis.percentage_of_population('education')
    v = my_analysis.get_count('education')['y']
    values = [v[1], v[0], v[2], v[3], v[4], v[5], v[6], v[7]]
    labels = ['basic_4y', 'basic_6y', 'basic_9y', 'high school', 'illiterate', 'professional course', 'university degree', 'unknown']
    fig = px.pie(percents,  values= values, names = labels, 
                title = '% of Population based on education')
    return fig

def education_level_prob():
    '''
    This function gives the plot of probability of success based on people's education level.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    data = my_analysis.get_probabilities('education')
    data['y'] = data['y']*100
    fig = px.bar(data, x='education', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'education': 'Education Level'},
                height=400, title = 'Probability of success by education')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def income_level_distribution():
    '''
    This function gives plot of distribution of people based on income level.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    percents = my_analysis.percentage_of_population('job')
    v = my_analysis.get_count('job')['y']
    values = [v[1], v[0], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]]
    labels = ['admin', 'blue-collar', 'entrepreneur', 'house maid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    fig = px.pie(percents,  values= values, names = labels, 
                title = '% of Population based on job')
    return fig

def job_prob():
    '''
    This function gives the plot of probability of success based on people's job level.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    job_prob = my_analysis.get_probabilities('job')
    data = job_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='job', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'job': 'Job'},
                height=400, title = 'Probability of success by job')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def contact_way_distribution():
    '''
    This function gives plot of distribution of people based on how they were contacted, i.e, cell phone or telephone.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    contact_count = my_analysis.get_count('contact')
    contact_success_count = my_analysis.get_success_count('contact')
    status=['cellular', 'telephone']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=contact_count['y']-contact_success_count['y']),
        go.Bar(name='Success', x=status, y=contact_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Contact type", yaxis_title="Number of people", title = 'Number of people contacted on cellular phone or telephone')
    return fig

def contact_prob():
    '''
    This function gives plot of probability of success based on how people were contacted, i.e, cell phone or telephone.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    contact_prob = my_analysis.get_probabilities('contact')
    data = contact_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='contact', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'contact': 'Contact type'},
                height=400, title = 'Probability of success by method of contact')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def loan_status():
    '''
    This function gives the plot of distribution of people based on loan status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    loan_count = my_analysis.get_count('loan')
    loan_success_count = my_analysis.get_success_count('loan')
    status=['yes', 'no', 'Info Not Available']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=loan_count['y']-loan_success_count['y']),
        go.Bar(name='Success', x=status, y=loan_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', title = "Do people have a loan?", xaxis_title = "Loan status", yaxis_title="Number of people", height=400)
    return fig

def loan_prob():
    '''
    This function gives plot of probability of success based on people's loan status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    loan_prob = my_analysis.get_probabilities('loan')
    data = loan_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='loan', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'loan': 'Loan status'},
                height=400, title = 'Probability of success by loan status')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def house_status_distribution():
    '''
      This function gives the plot of distribution of people based on housing status.

      Returns
      -------
      plotly.graph_objs._figure.Figure
          returns a interactive graph of marital status distribution.
          
          
      '''
    housing_count = my_analysis.get_count('housing')
    housing_success_count = my_analysis.get_success_count('housing')
    status=['yes', 'no', 'Info Not Available']

    fig = go.Figure(data=[
        go.Bar(name='Not Successful', x=status, y=housing_count['y']-housing_success_count['y']),
        go.Bar(name='Success', x=status, y=housing_success_count['y'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack', xaxis_title="Housing Status", yaxis_title="Number of people", height=400, title = 'Housing status of the population')
    return fig

def house_prob():
    '''
    This function gives plot of probability of success based on people's housing status.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    housing_prob = my_analysis.get_probabilities('housing')
    data = housing_prob
    data['y'] = data['y']*100
    fig = px.bar(data, x='housing', y='y',
                hover_data=data, labels={'y':'Probability of Success (%)', 'housing': 'Housig Status'},
                height=400, title = 'Probability of success by housing status')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

def prediction_pie_chart():
    '''
    Plot predicted telecaller success ratios on the test data.
    
    :return: Plots the prediction pie chart.
    :rtype: plotly.graph_objs._figure.Figure
    '''
    fig = px.pie(predictions, 
                 values=[1 for i in range(len(predictions))], 
                 names= np.where(predictions['pred'] == 1, 'Purchase', 'No Purchase'), 
                 title='Overall Telemarketing Success Predictions')
    return fig

def predicted_prob_hist():
    '''
    Plot the histogram of the predicted probabilities.
    
    :return: Plots the prediction probability histogram.
    :rtype: plotly.graph_objs._figure.Figure
    '''
    fig = px.histogram(predictions, 
                       x="prob_1", 
                       nbins=5, 
                       labels = {'prob_1' : 'Success Probabilty'},
                       title='Histogram of Success Probabilities')
    return fig

def age_distribution():
    '''
      This function gives the plot of distribution of people's responses based on age groups that they fall in.

      Returns
      -------
      plotly.graph_objs._figure.Figure
          returns a interactive graph of marital status distribution.
          
          
      '''
    No = [x[0] for x in myList]
    Yes = [x[1] for x in myList]
    x = np.arange(len(labels)) 
    width = 0.20  

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x - width/2,
        y = No,
        name='NO',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=x + width/2,
        y = Yes,
        name='YES',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        title = 'Count of yes/no response for different age groups',
        xaxis_title="Age Group",
        yaxis_title="Number of People",
        xaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(labels))],
            ticktext = labels
        )
    )
    return fig

def age_prob():
    '''
    This function gives the plot of distribution of people's responses based on age groups that they fall in.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        returns a interactive graph of marital status distribution.
        
        
    '''
    data = my_analysis.get_success_count("age")
    fig = px.bar(data, x='age', y='y',
                hover_data=data, labels={'y':'Number of Success (%)', 'age': 'Age'},
                height=400, title = 'Success for different ages')
    fig.update_traces(marker_color='#F8A19F', marker_line_color='rgb(111,64,112)',
    marker_line_width=1.5, opacity=0.8)
    return fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#app.scripts.config.serve_locally = True

app.config['suppress_callback_exceptions'] = True
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

tab_style = {
    'fontWeight': 'bold'
}
vis_tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '12px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '12px'
}
app.layout = html.Div(children = [
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Manager Dashboard', value='tab-1', style=tab_style),
        dcc.Tab(label='Telecaller Dashboard', value='tab-2', style=tab_style),
        dcc.Tab(label='Live Prediction Dashboard', value='tab-new', style=tab_style),
    ]),
    html.Div(id='tabs-content')
])
layout_tab_1  = html.Div(children = [
    dcc.Tabs(id = "vis-tabs", value = "vistab", vertical=True, parent_style={'float': 'left','width': '40'},children =[
        dcc.Tab(label='Marital Status', value='tab-3', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Educational Level', value='tab-4', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Income&Job', value='tab-5', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Contact Type', value='tab-6', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Loan Status', value='tab-7', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Housing Status', value='tab-8', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Age', value='tab-9', style=vis_tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Prediction Overview', value='tab-10', style=vis_tab_style, selected_style=tab_selected_style),
    ]),
    html.Div(id='vis-tabs-content',style={'float': 'right'})
])

marital_status_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = marital_state_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center' }),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = marital_status_probab()
            )],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])
educational_Level_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = education_level_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = education_level_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

income_vis = html.Div(children =[

            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = income_level_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = job_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

contact_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = contact_way_distribution()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = contact_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])

loan_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = loan_status()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = loan_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])
house_vis = html.Div(children =[            
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "marital status",
                figure = house_status_distribution()
            ) ],
            style={'height': 400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "marital prob",
                figure = house_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

])


prediction_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = prediction_pie_chart()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = predicted_prob_hist()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])

age_vis = html.Div(children =[
            html.Div([
            html.Div(children =[
                dcc.Graph(
                id = "prediction_pie_chart",
                figure = age_distribution()
            ) ],
            style={'height': 400,'width': '300', 'float': 'left', 'display': 'flex', 'justify-content': 'center'}),

            html.Div(children =[
                dcc.Graph(
                id = "predicted_prob_hist",
                figure = age_prob()
            )],
            style={'height':400,'width': '400', 'float': 'left', 'display': 'flex', 'justify-content': 'center'})
            ]),

        ])

@app.callback(Output('vis-tabs-content', 'children'),
              [Input('vis-tabs', 'value')])
def render_content(tab):
    if tab == 'tab-3':
        return marital_status_vis
    elif tab == 'tab-4':
        return educational_Level_vis
    elif tab == 'tab-5':
        return income_vis 
    elif tab == 'tab-6':
        return contact_vis 
    elif tab == 'tab-7':
        return loan_vis 
    elif tab == 'tab-8':
        return house_vis 
    elif tab == 'tab-9':
        return age_vis
    elif tab == "tab-10":
        return prediction_vis
    else:
        return marital_status_vis


layout_tab_2 = html.Div(children =[
             
             html.Div(dash_table.DataTable(
                         columns=[
                             {'name': 'Customer ID', 'id': 'customer_id', 'type': 'numeric', 'editable': False},
                             {'name': 'Age', 'id': 'age', 'type': 'numeric', 'editable': False},
                             {'name': 'Income', 'id': 'job_transformed', 'type': 'text', 'editable': False},
                             {'name': 'Previously Contacted', 'id': 'poutcome', 'type': 'text', 'editable': False},
                             {'name': 'Probability of Success', 'id': 'prob_1', 'type': 'numeric', 'editable': False, 'format': FormatTemplate.percentage(1)},
                             {'name': 'Call Result', 'id': 'is_called', 'type': 'any', 'editable': True, 'presentation': 'dropdown'}
                         ],
                         data=df.to_dict('records'),
                         filter_action='native',
                         dropdown={
                             'is_called': {
                                 'options': [
                                     {'label': i, 'value': i}
                                     for i in ['Not Called', 'Success', 'Failure']
                                 ]
                             }
                         },                
                         style_table={
                             'maxHeight': '50ex',
                             'overflowY': 'scroll',
                             'width': '100%',
                             'minWidth': '100%',
                         },
                         style_data={
                             'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
                             'overflow': 'hidden',
                             'textOverflow': 'ellipsis',
                         },
                         style_cell = {
                             'font_family': 'arial',
                             'font_size': '16px',
                             'text_align': 'center'
                         },
#                         style_cell_conditional=[
#                            {
#                                'if': {'column_id': c},
#                                'textAlign': 'left'
#                            } for c in ['customer_id', 'job_transformed', 'poutcome', 'is_called']
#                        ],
                         style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Not Called"'
                                },
                                'backgroundColor': '#E0E280'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Success"'
                                },
                                'backgroundColor': '#8CE280'
                            }, {
                                'if': {
                                    'column_id': 'is_called',
                                    'filter_query': '{is_called} eq "Failure"'
                                },
                                'backgroundColor': '#E28080'
                            }
                        ],
                         style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                         page_action="native",
                         page_current= 0,
                         sort_action="native",
                         sort_mode="multi"
                         )                    
                     )                
                
        ])

layout_tab_new = html.Div(children =[
    html.Div(children =[
    html.Div(children =[
    html.Label('Enter number of employees (quarterly indicator): '),
    dcc.Input(id='nremployed', placeholder='# employees', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the outcome of the previous marketing campaign: '),
    dcc.Input(id='poutcome_success', placeholder='prev.', type='number', min = 0, max = 1, step = 1)],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the employment variation rate - quarterly indicator: '),
    dcc.Input(id='emp', placeholder='emp. variation rate', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the number of days since the last call (999 if NA): '),
    dcc.Input(id='pdays', placeholder='# days since last call', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the consumer confidence index (monthly indicator): '),
    dcc.Input(id='consconfidx', placeholder='consumer conf. index', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the euribor 3 month rate (daily indicator): '),
    dcc.Input(id='euribor3m', placeholder='euribor rate', type='number')],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
              
    html.Div(children =[
    html.Label('Enter the no income indicator, 1 if the customer job retired, student or unemployed: '),
    dcc.Input(id='job_transformed_no_income', placeholder='inc', type='number', min = 0, max = 1, step = 1)],
              style={'float': 'center', 'display': 'flex', 'justify-content': 'center'}),
    
    ]),
   
    html.Div(children=[
        html.H1(children='Probability of Success: '),
        html.Div(id='pred-output')
    ], style={'textAlign': 'center', 'justify-content': 'center'}),
])
@app.callback(
    Output('pred-output', 'children'),
    [Input('nremployed', 'value'),
     Input('poutcome_success', 'value'),
     Input('emp', 'value'),
     Input('pdays', 'value'),
     Input('consconfidx', 'value'),
     Input('euribor3m', 'value'),
     Input('job_transformed_no_income', 'value')])
def show_success_probability(nr_employed, poutcome_success, emp_var_rate, pdays, cons_conf, euribor, no_income):
    if not nr_employed: 
        nr_employed = 0
    if not poutcome_success: 
        poutcome_success = 0
    if not emp_var_rate: 
        emp_var_rate = 0
    if not pdays: 
        pdays = 0
    if not cons_conf: 
        cons_conf = 0
    if not euribor: 
        euribor = 0
    if not no_income:
        no_income = 0
        
        #raise PreventUpdate
    #else:
    prob = dynamic_predict(nr_employed, poutcome_success, emp_var_rate, pdays, cons_conf, euribor, no_income)[0]*100
    return html.Div(children =[
        html.H1(children=f'{round(prob, ndigits = 2)}'+"%")
    ])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return layout_tab_1
    elif tab == 'tab-2':
        return layout_tab_2
    elif tab == "tab-new":
        return layout_tab_new

server = app.server

if __name__ == '__main__':
    model = joblib.load("LR_prediction.joblib")
    app.run_server(debug=True)
    #application.run_server(host='0.0.0.0', port=8050, debug=True)
    #application.run(debug=True, port=8080)
    #application.run_server(host='0.0.0.0')
    #app.run_server(host="0.0.0.0")