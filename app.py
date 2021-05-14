import os
import json
import random
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from util import plot_users, plot_network, plot_clusters, plot_closeness
from flask import render_template

def render_tab(name, res_dict):

    tab_style = {
        'padding-top': '15px',
        'height': 50, 
        'margin-top': 20, 
        'margin-bottom': 2, 
        'margin-left': 2, 
        'margin-right': 2, 
        'background-color': 'rgba(256, 256, 256, 0.8)',
        'border-style': 'solid',
        'border-radius': 7,
        'border-width': 0,
        'border-color': 'white',
        'overflow': 'hidden',
        'width': '45%'
    }

    tab_style_selected = {
        'padding-top': '15px',
        'align-items': 'center',
        'height': 50, 
        'margin-top': 20, 
        'margin-bottom': 2, 
        'margin-left': 2, 
        'margin-right': 2, 
        'background-color': '#ffffff',
        'border-style': 'solid',
        'border-radius': 7,
        'border-width': 2,
        'border-color': '#3b738f',
        'overflow': 'hidden',
        'width': '55%'
    }


    return dcc.Tab(
        label=name, 
        style=tab_style,
        selected_style=tab_style_selected, 
        children=[
            html.Div(className="div-btm", children=[
                html.Div(className="div-btm-inner", children=[
                    html.Div(className="div-network-graph", children=[
                        html.Div(className="div-graph-btm", children=[
                            html.Div(html.Span('Network Graph', className="graph-title"), className="div-graph-title"),
                            dcc.Graph(
                                className='graph-network',
                                config={'displayModeBar': False},
                                figure=plot_network(
                                    np.array(res_dict['adjacencym']), 
                                    res_dict['clusters'], 
                                    res_dict['cluster_names'], 
                                    {int(key): value for key, value in res_dict['id_to_name'].items()}, 
                                    seed)
                            )
                        ])
                    ]),

                    html.Div(className="div-mma-graph", children=[

                        html.Div(className="div-graph-btm", children=[

                            html.Div(html.Span('Avg-Max-Min', className="graph-title"), className="div-graph-title"),

                            dcc.Graph(
                                className='graph-mma',
                                config={'displayModeBar': False},
                                figure=plot_clusters(
                                    res_dict['cluster_size'], 
                                    res_dict['cluster_max'], 
                                    res_dict['cluster_min'], 
                                    res_dict['cluster_avg'], 
                                    res_dict['cluster_names'])
                            )
                        ])

                    ]),
                    html.Div(className="div-heatmap-graph", children=[
                        html.Div(className="div-graph-btm", children=[
                            html.Div(className="div-graph-btm", children=[
                                html.Div(html.Span('Closeness Between Clusters', className="graph-title"), className="div-graph-title"),
                                dcc.Graph(
                                    className='graph-heatmap',
                                    config={'displayModeBar': False},
                                    figure=plot_closeness(
                                        res_dict['closeness'])
                                )
                            ])
                        ])
                    ])
                ])
        ])
])


app = dash.Dash(__name__)
server = app.server
app.title = 'Network Analysis'

with open(os.path.join('output', 'cluster1.json')) as f:
    group1 = json.load(f)

with open(os.path.join('output', 'cluster2.json')) as f:
    group2 = json.load(f)

user_stats = pd.read_csv(os.path.join('output', 'user_stats.csv'), header=0, index_col=0)
user_stats.columns = [0, 1, 2, 3]

seed = random.randint(1, 100)

app.layout = html.Div([
    html.Div(className="div-top", children=[
        html.Div(className="div-top-inner", children=[
            html.Div(className="div-desc", children=[
                html.Div(className="div-title", 
                    children=html.Span(className="title", children='Network Analysis')
                ),
                html.Div(className="div-desc-inner", children=[
                    html.Div(className="links", children=[
                        html.A(
                            '',
                            href='#',
                        ),
                        html.A(
                            '', 
                            href='#',
                            target="_blank"
                        ),
                    ]),
                    html.Div(className="caption", children="On the plot, click and drag to zoom in, double click to rescale."),
                    html.Div(className="caption", children="On the legend, click or double click to filter."),
                    html.Div(className="caption", children='Usernames in the network are masked with randomly generated string.')
                ])
            ]),

            html.Div(className="div-user-graph", children=[
                html.Div(className="div-graph-top", children=[
                    html.Div(html.Span('Percentage of Mutual Connection', className="graph-title"), className="div-graph-title"),
                        dcc.Graph(
                            className='graph-user',
                            config={'displayModeBar': False},
                            figure=plot_users(user_stats)
                        )   
                ])         
            ])
        ])
    ]),

    dcc.Tabs(
        parent_className='custom-tabs',
        className='custom-tabs-container', 
        children=[
            render_tab('No. of clusters = 8', group1),
            render_tab('No. of clusters = 16', group2)
    ])

])

if __name__ == '__main__':
    app.run_server(debug=True)
