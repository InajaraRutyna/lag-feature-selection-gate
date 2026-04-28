# display_graph.py

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import io
import logging
import os
from config.load import *  # Import load_config to access config.yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
plots_dir = os.path.abspath(config['datasets']['plots_dir'])

def get_graphs_modal():
    """
    Returns the Modal layout containing graph controls and display areas.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader("Variable Graphs"),
            dbc.ModalBody(
                [
                    # Variable Selection
                    html.Div([
                        html.Label("Select Variables to Visualize:", className="control-label"),
                        dcc.Dropdown(
                            id='selected-variables-dropdown',
                            options=[],  # To be populated dynamically
                            value=[],  # Initially no selection
                            multi=True,
                            placeholder="Select one or more variables",
                            className="dropdown_graph",
                            style={'width': '100%'}
                        ),
                    ], className="control-group", style={'margin-bottom': '20px'}),

                    # Graph Type Selection
                    html.Div([
                        html.Label("Select Graph Types:", className="control-label"),
                        dcc.Checklist(
                            id='selected-graph-types',
                            options=[
                                {'label': 'Line Chart', 'value': 'line'},
                                {'label': 'Scatter Plot', 'value': 'scatter'},
                                {'label': 'Box Plot', 'value': 'box'},
                                {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                                {'label': 'Moving Average', 'value': 'ma'},
                            ],
                            value=['line'],  # Default selection
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                            className="checkbox_graph"
                        ),
                    ], className="control-group", style={'margin-bottom': '20px'}),

                    # Color Selection with Radio Buttons
                    html.Div([
                        html.Label("Select Plot Color:", className="control-label"),
                        dcc.RadioItems(
                            id='plot-color-radio',
                            options=[
                                {'label': 'Blue', 'value': 'blue'},
                                {'label': 'Red', 'value': 'red'},
                                {'label': 'Green', 'value': 'green'},
                                {'label': 'Orange', 'value': 'orange'},
                                {'label': 'Purple', 'value': 'purple'},
                                {'label': 'Black', 'value': 'black'},
                            ],
                            value='blue',  # Default color
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                            className="radio_plot_color"
                        ),
                    ], className="control-group", style={'margin-bottom': '20px'}),

                    # Point Size Selection for Scatter Plot
                    html.Div([
                        html.Label("Select Point Size for Scatter Plot:", className="control-label"),
                        dcc.RadioItems(
                            id='scatter-size-radio',
                            options=[
                                {'label': 'Very Small', 'value': 5},
                                {'label': 'Small', 'value': 10},
                                {'label': 'Medium', 'value': 15},
                                {'label': 'Large', 'value': 20},
                                {'label': 'Extra Large', 'value': 25},
                            ],
                            value=10,  # Default size
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                            className="radio_scatter_size"
                        ),
                    ], className="control-group", style={'margin-bottom': '20px'}),

                    # Graph Display Area
                    html.Div([
                        dcc.Loading(
                            id="loading-graphs",
                            type="default",
                            children=html.Div(id='graphs-container')
                        )
                    ], className="graphs-container"),
                ]
            ),
            dbc.ModalFooter(
                [
                    # Left side: Filename input and Save button
                    html.Div([
                        dbc.Input(
                            id="save-filename",
                            type="text",
                            placeholder="Enter filename (e.g., my_plot.png)",
                            value="my_plot.png",
                            style={'width': '70%', 'display': 'inline-block', 'margin-right': '10px'}
                        ),
                        dbc.Button("Save Plot", id="save-plot-button", color="primary", className="mr-2"),
                        html.Div(id="save-feedback_g", style={'display': 'inline-block', 'margin-left': '10px'}),
                    ], style={'float': 'left', 'display': 'inline-block'}),
                    # Right side: Close button
                    dbc.Button("Close", id="close-graphs-modal", color="secondary", style={'float': 'right'})
                ]
            ),
        ],
        id="graphs-modal",
        is_open=False,
        size="xl",
        centered=True,
        className="graphs-modal"  # For custom styling
    )

def register_callbacks(app):
    """
    Registers the necessary callbacks for the graphs Modal, including dynamic graph selection and updates.
    """

    @app.callback(
        [Output("graphs-modal", "is_open"),
         Output('selected-variables-dropdown', 'options')],
        [Input("open-graphs-modal", "n_clicks"),
         Input("close-graphs-modal", "n_clicks")],
        [State("graphs-modal", "is_open"),
         State('processed-data-store', 'data')]
    )
    def toggle_modal(n_open, n_close, is_open, processed_data_store):
        """
        Toggles the Modal open/close state based on button clicks and populates variable options.
        """
        ctx = callback_context

        if not ctx.triggered:
            return is_open, dash.no_update

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == "open-graphs-modal" and n_open:
            logger.info("Opening graphs Modal.")
            # Populate variable options based on processed_data_store
            if processed_data_store:
                try:
                    # Assuming processed_data_store is a JSON string or a dictionary with 'data' key
                    if isinstance(processed_data_store, dict):
                        data_json = processed_data_store.get('data')
                    else:
                        data_json = processed_data_store
                    data = pd.read_json(io.StringIO(data_json), orient='split')

                    # Identify numerical columns
                    numeric_cols = data.select_dtypes(include=['number']).columns

                    variable_options = [{'label': col, 'value': col} for col in numeric_cols]
                    logger.info(f"Available numerical variables for visualization: {numeric_cols.tolist()}")

                except Exception as e:
                    logger.error(f"Error parsing processed_data_store: {e}")
                    variable_options = []
            else:
                variable_options = []

            return True, variable_options

        elif triggered_id == "close-graphs-modal" and n_close:
            logger.info("Closing graphs Modal.")
            return False, dash.no_update

        return is_open, dash.no_update

    @app.callback(
        Output('graphs-container', 'children'),
        [Input('selected-variables-dropdown', 'value'),
         Input('selected-graph-types', 'value'),
         Input('plot-color-radio', 'value'),
         Input('scatter-size-radio', 'value')],
        [State('processed-data-store', 'data')]
    )
    def update_graphs(selected_variables, selected_graph_types,
                      plot_color, scatter_point_size, processed_data_store):
        """
        Generates and displays graphs based on selected variables and graph types.
        Also handles color and size customizations.
        """
        ctx = callback_context

        triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Only proceed if triggered by relevant inputs
        if triggered not in ['selected-variables-dropdown', 'selected-graph-types',
                             'plot-color-radio', 'scatter-size-radio']:
            return dash.no_update

        if not processed_data_store:
            logger.warning("No data available in processed-data-store.")
            return dbc.Alert("No data available to display graphs.", color="warning")

        if not selected_variables:
            return dbc.Alert("Please select at least one variable to visualize.", color="info")

        if not selected_graph_types:
            return dbc.Alert("Please select at least one graph type.", color="info")

        try:
            # Load the processed data
            if isinstance(processed_data_store, dict):
                data_json = processed_data_store.get('data')
            else:
                data_json = processed_data_store
            data = pd.read_json(io.StringIO(data_json), orient='split')
            logger.info(f"Generating graphs for variables: {selected_variables} with types: {selected_graph_types}")

            graphs = []

            for var in selected_variables:
                if 'line' in selected_graph_types:
                    fig_line = px.line(
                        data,
                        x=data.index,
                        y=var,
                        title=f"Line Chart of {var}",
                        template="plotly_white",
                        color_discrete_sequence=[plot_color]
                    )
                    graphs.append(dcc.Graph(figure=fig_line, config={'displayModeBar': False}, id={'type': 'dynamic-graph', 'index': f'line-{var}'}))

                if 'scatter' in selected_graph_types:
                    fig_scatter = px.scatter(
                        data,
                        x=data.index,
                        y=var,
                        title=f"Scatter Plot of {var}",
                        template="plotly_white",
                        color_discrete_sequence=[plot_color]
                    )
                    # Update marker size
                    if scatter_point_size:
                        fig_scatter.update_traces(marker=dict(size=scatter_point_size))
                    graphs.append(dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}, id={'type': 'dynamic-graph', 'index': f'scatter-{var}'}))

                if 'box' in selected_graph_types:
                    fig_box = px.box(
                        data,
                        y=var,
                        title=f"Box Plot of {var}",
                        template="plotly_white",
                        color_discrete_sequence=[plot_color]
                    )
                    graphs.append(dcc.Graph(figure=fig_box, config={'displayModeBar': False}, id={'type': 'dynamic-graph', 'index': f'box-{var}'}))

                if 'ma' in selected_graph_types:
                    # Generate Moving Average Plot
                    ma_series = data[var].rolling(window=30).mean()
                    fig_ma = px.line(
                        x=data.index,
                        y=ma_series,
                        title=f"30-Day Moving Average of {var}",
                        labels={'x': 'Date', 'y': var},
                        template="plotly_white",
                        color_discrete_sequence=[plot_color]
                    )
                    graphs.append(dcc.Graph(figure=fig_ma, config={'displayModeBar': False}, id={'type': 'dynamic-graph', 'index': f'ma-{var}'}))

            # Correlation Heatmap (only once, not per variable)
            if 'heatmap' in selected_graph_types:
                corr = data[selected_variables].corr()
                fig_heatmap = px.imshow(
                    corr,
                    text_auto=True,
                    title="Correlation Heatmap",
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    template="plotly_white"
                )
                graphs.append(dcc.Graph(figure=fig_heatmap, config={'displayModeBar': False}, id={'type': 'dynamic-graph', 'index': 'heatmap'}))

            if not graphs:
                return dbc.Alert("No graphs to display based on your selections.", color="warning")

            return html.Div(graphs, className="graphs-wrapper")

        except Exception as e:
            logger.error(f"Error generating graphs: {e}")
            return dbc.Alert(f"An error occurred while generating graphs: {e}", color="danger")

    @app.callback(
        Output('save-feedback_g', 'children'),
        Input('save-plot-button', 'n_clicks'),
        [State('graphs-container', 'children'),
         State('save-filename', 'value')]
    )
    def save_plots(n_clicks, graphs_children, filename):
        """
        Saves the currently displayed plots to the specified directory with the given filename.
        """
        if n_clicks is None or n_clicks == 0:
            return dash.no_update

        if not graphs_children:
            logger.warning("No graphs to save.")
            return dbc.Alert("No graphs to save.", color="warning")

        if not filename:
            logger.warning("No filename provided.")
            return dbc.Alert("Please enter a filename.", color="warning")

        # Ensure the filename ends with .png
        if not filename.lower().endswith('.png'):
            filename += '.png'

        # Prepare the save path
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        save_path = os.path.join(plots_dir, filename)

        try:
            # Find all the figures from the graphs
            figures = []
            for child in graphs_children:
                if isinstance(child, dict) and 'props' in child and 'figure' in child['props']:
                    fig = child['props']['figure']
                    figures.append(fig)
                elif hasattr(child, 'props') and 'figure' in child.props:
                    fig = child.props['figure']
                    figures.append(fig)

            if not figures:
                logger.warning("No figures found in graphs.")
                return dbc.Alert("No figures found to save.", color="warning")

            # For simplicity, save the first figure (since saving multiple figures to one image is complex)
            # Alternatively, you can create a subplot if needed
            fig_to_save = figures[0]
            fig_bytes = fig_to_save.to_image(format="png")

            with open(save_path, 'wb') as f:
                f.write(fig_bytes)
            logger.info(f"Plot saved to {save_path}")
            return dbc.Alert(f"Plot saved to {save_path}", color="success")

        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return dbc.Alert(f"An error occurred while saving the plot: {e}", color="danger")
