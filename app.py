import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import base64
import os

from main import load_config
import code_visual.train_mode as train_mode
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize the Dash app with Bootstrap CSS
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, '/assets/style.css', '/assets/train.css'],
                suppress_callback_exceptions=True)
app.title = "Time-Series Pipeline"

# Register train mode callbacks
train_mode.register_callbacks(app)  # Ensure train_mode callbacks are registered here

# Function to encode images
def encode_image(image_path):
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
    return f"data:image/png;base64,{encoded_image}"

def create_radio_items(id, options, value):
    return dcc.RadioItems(
        id=id,
        options=[{'label': opt, 'value': opt} for opt in options],
        value=value,
        className="custom-radio",
        labelStyle={'display': 'block', 'margin-bottom': '10px', 'color': 'white'}
    )

# Function to get dynamic header content
def get_header(mode):
    title = "Train Mode - Pipeline Configuration"
    description = "Train Mode: Set up your model training pipeline."


    return html.Div(
        [
            html.H4(title, className="header"),
            html.P(description, className="subheader"),
        ],
        className="header-container",
    )

# Layout for the Sidebar
sidebar = html.Div(
    [
        html.H5("📈 Time-Series ML Pipeline", className="display-5"), html.Hr(), html.Br(),
        # Sidebar content
        html.Div(
            [
                # Renewable energy selection
                html.H5("Renewable Energy:", style={'color': 'white'}),
                create_radio_items(id='data_type', options=['Wind', 'Solar'], value='Wind'),

                dcc.Store(id='train_or_test', data='Train'),

                # Corresponding image based on data_type (Wind or Solar)
                html.Div(id='image-display', className="sidebar-image"),
            ],
            className="sidebar-content",
            style={"flex": "1", "paddingBottom": "20px"}
        ),
        html.Div(
            [
                html.Div(id='footer-data-type', className="footer-line-1"),  # First line for Wind/Solar data selection
                html.Div(id='footer-mode', className="footer-line-2")  # Second line for Train message
            ],
            className="sidebar-footer"
        )
    ],
    className="sidebar"
)

# Main content layout
content = html.Div(
    [
        dcc.Store(id='data-type-store', storage_type='session'), # Hidden Store to hold the data_type
        html.Div(id="top-right-image"),  # Placeholder for the top-right image
        # Dynamic header based on mode
        html.Div(id="dynamic-header", className="header-container"),
        html.Div(id="main-content"),  # This will change based on `train_or_test`
        # Placeholders for components that will be dynamically generated
        html.Div(id="train-content", style={'display': 'none'}),  # Hidden placeholder for train content
    ],
    className="main-content"
)

app.layout = dbc.Container(
    dbc.Row(
        [
            dbc.Col(sidebar, width=2),  # Sidebar occupies 2/12 columns (roughly 15%)
            dbc.Col(content, width=10)  # Main content occupies 10/12 columns (roughly 85%)
        ],
        className='app-content'
    ),
    fluid=True,  # Ensure the container is fluid (full-width)
    style={'height': '100vh'}  # Full viewport height
)

@app.callback(Output('image-display', 'children'),
              Input('data_type', 'value'))
def update_image(data_type):
    # Set the path to the image based on the selection (Wind or Solar)
    image_path = f"./fig/{data_type}.png"
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
    # Return the image HTML component
    return html.Img(src=f"data:image/png;base64,{encoded_image}", className="sidebar-image")

# Callback for updating the first line of the footer based on data_type
@app.callback(Output('footer-data-type', 'children'),
              Input('data_type', 'value'))
def update_footer_data_type(data_type):
    if data_type == 'Wind':
        return "Wind Data Selected"
    elif data_type == 'Solar':
        return "Solar Data Selected"

# Callback to update the second line of the footer based on train/test mode
@app.callback(Output('footer-mode', 'children'),
              Input('train_or_test', 'value'))
def update_footer_mode(train_or_test):
    return "Train new ML Model"


# Callback to store the selected data_type in dcc.Store
@app.callback(Output('data-type-store', 'data'),
              Input('data_type', 'value'))
def update_data_type_store(selected_data_type):
    return selected_data_type

# Callback to update the page content based on the 'train_or_test' selection
@app.callback([Output('dynamic-header', 'children'),
               Output('main-content', 'children'),
               Output('top-right-image', 'children')],
              [Input('train_or_test', 'value')])

def update_main_content(train_or_test):
    # Get the dynamic header based on the mode
    header = get_header(train_or_test)

    # Train mode content
    content = train_mode.get_train_mode_layout()  # Use the function from train_mode
    # Set the train image
    img_src = encode_image("./fig/Train.png")

    # Return the updated header, content, and image for the top-right corner
    top_right_image = html.Img(src=img_src, className="top-right-image")
    return header, content, top_right_image

# Main function to run the app
if __name__ == '__main__':
    app.run(debug=False)
