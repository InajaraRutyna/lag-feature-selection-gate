from dash import dcc, html, Input, Output, State, callback_context, no_update
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import os
import io
import re
import gc
import time
import logging
import threading
import contextlib
import traceback
from queue import Queue, Empty

import pandas as pd

from main import main as run_single_model, zero_weeks_to_nan
from run_all import save_metrics_to_excel
import code_ai.shared_variables as sv
from code_ai.data_treatment import load_data, handle_categorical_columns
from config.load import *
import code_visual.display_graph as display_graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

RAW_DATASETS_DIR = os.path.abspath(config["datasets"]["raw_datasets_dir"])
INSERTED_DATASETS_DIR = os.path.abspath(config["datasets"]["inserted_datasets_dir"])


RUN_STATE = {
    "running": False,
    "done": False,
    "error": None,
    "result": None,
    "logs": [],
    "queue": Queue(),
    "thread": None,
}


class QueueWriter:
    def __init__(self, queue):
        self.queue = queue

    def write(self, message):
        if message and message.strip():
            self.queue.put(message.rstrip())

    def flush(self):
        pass

class QueueLogHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        try:
            message = self.format(record)
            if message and message.strip():
                self.queue.put(message.rstrip())
        except Exception:
            self.handleError(record)


def reset_run_state():
    RUN_STATE["running"] = False
    RUN_STATE["done"] = False
    RUN_STATE["error"] = None
    RUN_STATE["result"] = None
    RUN_STATE["logs"] = []
    RUN_STATE["queue"] = Queue()
    RUN_STATE["thread"] = None


def drain_run_logs():
    while True:
        try:
            message = RUN_STATE["queue"].get_nowait()
        except Empty:
            break

        RUN_STATE["logs"].append(message)

    max_lines = 600
    if len(RUN_STATE["logs"]) > max_lines:
        RUN_STATE["logs"] = RUN_STATE["logs"][-max_lines:]

    return "\n".join(RUN_STATE["logs"])


def has_complete_pipeline_metadata(metadata):
    required_fields = [
        "input_features",
        "output_variable",
        "feature_filtering",
        "model",
    ]

    return all(metadata.get(field) for field in required_fields)


def has_valid_categorization(metadata):
    method = metadata.get("categorization_method")
    return method not in [None, "", "Not Specified"]


def process_data_and_get_info(dataset_path):
    try:
        if os.path.isdir(dataset_path):
            all_files = [
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f))
            ]

            if not all_files:
                raise Exception("No data files found in the selected folder.")

            r_data = load_data(all_files)
            metadata = {}

        elif os.path.isfile(dataset_path):
            with open(dataset_path, "rb") as f:
                saved_data = pd.read_pickle(f)

            r_data = saved_data.get("data", None)
            metadata = saved_data.get("metadata", {})

            if metadata.get("categorization_method") == "Not Specified":
                metadata["categorization_method"] = None

        else:
            raise Exception("Invalid dataset path.")

        if r_data is None:
            raise Exception("Failed to load data.")

        return r_data, metadata

    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return None, None


def get_dataframe_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    lines = info.split("\n")
    lines_to_keep = []

    for line in lines:
        if "<class" in line:
            continue
        if "memory usage" in line:
            continue
        if "dtypes:" in line:
            continue
        lines_to_keep.append(line)

    return "\n".join(lines_to_keep)


def generate_html_table(dataframe, suffix, max_rows=10):
    dataframe = dataframe.head(n=max_rows)
    dataframe = dataframe.reset_index().rename({"index": "Index"}, axis="columns")

    return html.Div(
        [
            html.H5(f"Dataframe Head{suffix}:", className="header_1"),
            dbc.Table(
                [html.Thead(html.Tr([html.Th(col) for col in dataframe.columns]))]
                + [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(dataframe.iloc[i][col])
                                    for col in dataframe.columns
                                ]
                            )
                            for i in range(len(dataframe))
                        ]
                    )
                ],
                className="table_info_t",
                bordered=True,
                hover=True,
                responsive=True,
                striped=True,
            ),
        ],
        className="dataframe-head",
    )


def sanitize_filename(filename):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(filename))


def display_dataset_dropdown():
    return html.Div(
        [
            html.H5("Select Dataset Option:", className="header_1"),
            dcc.Dropdown(
                id="dataset-option-dropdown",
                options=[
                    {"label": "New Dataset", "value": "new_dataset"},
                    {"label": "Select Dataset from Folder", "value": "select_from_folder"},
                ],
                value=None,
                placeholder="Select a dataset option",
                className="dropdown_t",
            ),
            html.Div(id="dataset-option-content"),
        ],
        className="dropdown-container_t",
    )


def display_dataset_selection_feedback():
    return html.Div(
        [
            html.Div(
                [
                    html.H5("Select Dataset:", className="header_1"),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        options=[],
                        value=None,
                        placeholder="Select a dataset",
                        className="dropdown_t",
                    ),
                ],
                id="dataset-dropdown-container",
                className="dropdown-container_t",
                style={"display": "none"},
            ),
            html.Div(id="dataset-selection-feedback"),
        ],
        className="dropdown-container_t",
    )


def display_categorical_options():
    return html.Div(
        [
            html.Div(id="categorical-options-container"),
            html.Br(),
            html.Div(
                [
                    html.H5("Select Categorical Method:", className="header_1"),
                    dcc.Dropdown(
                        id="categorization-method-dropdown",
                        options=[],
                        value=None,
                        placeholder="Select a categorical method",
                        className="dropdown_larger_1",
                    ),
                    html.Br(),
                    html.Button(
                        "Confirm Categorical Choices",
                        id="confirm-categorical-choices",
                        n_clicks=0,
                        className="confirm-button",
                    ),
                ],
                id="categorization-method-container",
                style={"display": "none"},
            ),
            html.Br(),
            html.Div(
                id="metadata-display",
                className="metadata-display",
                style={"display": "block"},
            ),
        ],
        className="dropdown-container_t",
    )


def display_input_output_variables():
    return html.Div(
        [
            html.Br(),
            html.H5("Define Input Features and Output Variable:", className="header_1"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Features", className="header_2"),
                            dbc.Checklist(
                                options=[{"label": "Select All", "value": "ALL"}],
                                value=[],
                                id="select-all-features",
                                switch=True,
                                className="select-all-checkbox",
                            ),
                            dcc.Checklist(
                                id="input-features-checkbox",
                                options=[],
                                value=[],
                                className="feature-checklist",
                                labelStyle={
                                    "display": "block",
                                    "margin-bottom": "5px",
                                    "color": "white",
                                },
                            ),
                        ],
                        width="auto",
                        style={"flex": "0 1 auto", "padding": "0 5px"},
                    ),
                    dbc.Col(
                        [
                            html.H4("Output", className="header_2"),
                            dcc.RadioItems(
                                id="output-variable-radio",
                                options=[],
                                value=None,
                                className="output-radio",
                                labelStyle={
                                    "display": "block",
                                    "margin-bottom": "5px",
                                    "color": "white",
                                },
                            ),
                        ],
                        width="auto",
                        style={"flex": "0 1 auto", "padding": "0 5px"},
                    ),
                ],
                className="input-output-row",
            ),
            html.Br(),
            html.H5("Model Configuration:", className="header_1"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Select a Model:", className="header_2"),
                            dbc.Checklist(
                                options=[{"label": "Select All", "value": "ALL"}],
                                value=[],
                                id="select-all-models",
                                switch=True,
                                className="select-all-checkbox",
                            ),
                            dcc.Checklist(
                                id="models-checkbox",
                                options=[],
                                value=[],
                                className="feature-checklist",
                                labelStyle={
                                    "display": "block",
                                    "margin-bottom": "5px",
                                    "color": "white",
                                },
                            ),
                        ],
                        width="auto",
                        style={"flex": "0 1 auto", "padding": "0 5px"},
                    ),
                    dbc.Col(
                        [
                            html.H4("Select a Feature Filtering Method:", className="header_2"),
                            dcc.RadioItems(
                                id="feature-filtering-radio",
                                options=[],
                                value=None,
                                className="output-radio",
                                labelStyle={
                                    "display": "block",
                                    "margin-bottom": "5px",
                                    "color": "white",
                                },
                            ),
                        ],
                        width="auto",
                        style={"flex": "0 1 auto", "padding": "0 5px"},
                    ),
                ],
                className="input-output-row",
            ),
            html.Div(id="input-output-feedback", className="input-output-feedback"),
        ],
        id="input-output-variables-container",
        style={"display": "none"},
    )


def display_save_container():
    return html.Div(
        [
            html.H5("Save new pipeline configuration:", className="header_1"),
            dcc.Input(
                id="file-name-input",
                type="text",
                placeholder="Enter file name",
                value="",
                className="file-name-input_t",
            ),
            html.Button(
                "Save Data",
                id="save-data-button",
                n_clicks=0,
                className="save-button_t",
            ),
            html.Div(id="save-feedback", className="save-feedback"),
        ],
        id="save-container_t",
        className="save-container_t",
        style={"display": "none"},
    )


def display_variables_graphs():
    return html.Div(
        [
            dbc.Button(
                "Show Graphs",
                id="open-graphs-modal",
                n_clicks=0,
                className="show-graphs-button",
                color="primary",
                style={"margin-top": "20px"},
            ),
            display_graph.get_graphs_modal(),
        ],
        id="graphs-container_t",
        className="graphs-section",
        style={"display": "none"},
    )


def run_pipeline():
    return html.Div(
        [
            html.H5("Run Configuration:", className="header_1"),

            html.H5("Excel file name:", className="header_2"),
            dcc.Input(
                id="run-excel-file-name",
                type="text",
                placeholder="Leave empty for automatic file name",
                value="",
                className="file-name-input_t",
            ),

            html.Button(
                "Run Pipeline",
                id="run-pipeline-button",
                n_clicks=0,
                className="run-pipeline-button",
            ),

            dcc.Interval(
                id="run-log-interval",
                interval=1000,
                n_intervals=0,
                disabled=True,
            ),
            dcc.Store(id="run-terminal-user-scrolled", data=False),
            html.Div(
                html.Pre(
                    id="run-terminal-output",
                    children="",
                    style={
                        "backgroundColor": "#111",
                        "color": "#d6d6d6",
                        "padding": "12px",
                        "height": "420px",
                        "overflowY": "auto",
                        "overflowX": "auto",
                        "whiteSpace": "pre",
                        "border": "1px solid #444",
                        "borderRadius": "6px",
                        "marginTop": "15px",
                        "maxWidth": "100%",
                        "display": "block",
                    },
                ),
                id="run-terminal-wrapper",
            ),

            html.Div(id="run-pipeline-output", className="run-pipeline-output"),
        ],
        id="run-pipeline-container",
        style={"display": "none"},
    )


def get_train_mode_layout():
    return html.Div(
        [
            dcc.Store(id="processed-data-store"),
            html.Div(
                [
                    html.H2("Model Configuration", className="header_1"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    display_dataset_dropdown(),
                                    display_dataset_selection_feedback(),
                                    display_categorical_options(),
                                ],
                                width=4,
                            ),
                            dbc.Col(width=1),
                            dbc.Col(
                                dcc.Loading(
                                    id="loading-data-processing",
                                    type="default",
                                    children=html.Div(
                                        id="dataframe-info",
                                        className="dataframe-info",
                                        style={"display": "none"},
                                    ),
                                ),
                                width=7,
                            ),
                        ]
                    ),
                    html.Br(),
                    html.Div(
                        dbc.Col(
                            id="dataframe-head",
                            className="dataframe-head",
                            style={"display": "none"},
                            width=12,
                        )
                    ),
                    display_variables_graphs(),
                    display_input_output_variables(),
                    dbc.Row(
                        [
                            dbc.Col([], width=4),
                            dbc.Col(width=1),
                            dbc.Col(display_save_container()),
                        ]
                    ),
                    run_pipeline(),
                ],
                className="train-mode-content",
            ),
        ],
        className="app-content_t",
    )


def prepare_dash_data_for_main(data, metadata):
    run_metadata = metadata.copy()

    input_features = run_metadata.get("input_features", [])
    output_variable = run_metadata.get("output_variable", None)

    if not input_features or output_variable is None:
        raise ValueError("Input features or output variable missing from metadata.")

    input_features = list(input_features)

    if output_variable not in input_features:
        input_features.append(output_variable)

    data_input = data[input_features].copy()

    new_output = f"output_{output_variable}"
    data_input[new_output] = zero_weeks_to_nan(data_input[output_variable])

    data_input = data_input.groupby(level=0).mean()

    run_metadata["input_features"] = input_features
    run_metadata["output_variable"] = new_output

    sv.metadata = run_metadata
    sv.data_input = data_input.astype("float32")

    sv.data_out = None
    sv.seq_reduction = None
    sv.is_sequence = False


    return run_metadata

def is_solar_dataset(dataset_name):
    if not dataset_name:
        return False

    dataset_name = str(dataset_name).lower()
    return "solar" in dataset_name or "pv" in dataset_name or "fv" in dataset_name

def run_models_worker(data_json, metadata, dataset_name, excel_file_name):
    optuna_logger = None
    old_optuna_handlers = None
    old_optuna_level = None
    old_optuna_propagate = None

    try:
        RUN_STATE["queue"].put("Starting pipeline run...")

        data = pd.read_json(io.StringIO(data_json), orient="split")

        dash_metadata = metadata.copy()
        prepared_metadata = prepare_dash_data_for_main(data, dash_metadata)

        sv.SOLAR = is_solar_dataset(dataset_name)
        RUN_STATE["queue"].put(f"Solar dataset: {sv.SOLAR}")

        selected_models = prepared_metadata.get("model", [])
        feature_reduction = prepared_metadata.get("feature_filtering", None)

        if isinstance(selected_models, str):
            selected_models = [selected_models]

        if not selected_models:
            raise ValueError("No models selected.")

        if not feature_reduction:
            raise ValueError("No feature filtering method selected.")

        RUN_STATE["queue"].put(f"Selected models: {selected_models}")
        RUN_STATE["queue"].put(f"Selected input features: {prepared_metadata.get('input_features')}")
        RUN_STATE["queue"].put(f"Selected output variable: {prepared_metadata.get('output_variable')}")
        RUN_STATE["queue"].put(f"Selected feature filtering: {feature_reduction}")
        RUN_STATE["queue"].put(f"Feature creation: {prepared_metadata.get('features_creation')}")

        model_config = load_model_config()
        timesteps = model_config["prediction_config"]["timesteps"]
        steps_ahead = model_config["prediction_config"]["steps_ahead"]

        clean_dataset_name = sanitize_filename(dataset_name or "dataset")
        clean_feature_reduction = sanitize_filename(feature_reduction)

        excel_file_name = (excel_file_name or "").strip()
        clean_excel_file_name = sanitize_filename(excel_file_name)

        default_output_name = (
            f"{clean_dataset_name}_{timesteps}_ahead_"
            f"{steps_ahead}_{clean_feature_reduction}"
        )

        if clean_excel_file_name:
            output_name = clean_excel_file_name
        else:
            output_name = default_output_name

        sv.OUTPUT_DIR = os.path.join("tests", output_name)
        os.makedirs(sv.OUTPUT_DIR, exist_ok=True)

        excel_filename = os.path.join(sv.OUTPUT_DIR, f"{output_name}.xlsx")

        completed_models = []

        stdout_writer = QueueWriter(RUN_STATE["queue"])
        stderr_writer = QueueWriter(RUN_STATE["queue"])

        queue_log_handler = QueueLogHandler(RUN_STATE["queue"])
        queue_log_handler.setLevel(logging.INFO)
        queue_log_handler.setFormatter(
            logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s")
        )

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.INFO)

            optuna_logger = logging.getLogger("optuna")
            old_optuna_handlers = optuna_logger.handlers[:]
            old_optuna_level = optuna_logger.level
            old_optuna_propagate = optuna_logger.propagate

            for handler in old_optuna_handlers:
                optuna_logger.removeHandler(handler)

            optuna_logger.addHandler(queue_log_handler)
            optuna_logger.setLevel(logging.INFO)
            optuna_logger.propagate = False

        except Exception as e:
            RUN_STATE["queue"].put(f"Optuna logging capture setup failed: {e}")

        with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
            for model_type in selected_models:
                print(f"\n=== Running model: {model_type} ===")
                print(f"Using feature filtering method: {feature_reduction}")
                start_time = time.time()

                (
                    average_dataset_error,
                    train_error_all,
                    dataset_metrics_all,
                    validation_error_all,
                    best_params,
                    param_summary,
                ) = run_single_model(
                    model=model_type,
                    feature_filtering_method=feature_reduction,
                    size_data="All",
                )

                elapsed_time = time.time() - start_time
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)

                run_time = "{:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours),
                    int(minutes),
                    seconds,
                )

                if best_params:
                    if param_summary is None:
                        param_summary = {}

                    param_summary["run_time"] = {
                        "min": run_time,
                        "max": "",
                        "type": "info",
                    }
                else:
                    best_params = {"run_time": run_time}

                save_metrics_to_excel(
                    average_dataset_error=average_dataset_error,
                    average_model_error=train_error_all.reset_index(drop=True),
                    dataset_metrics_all=dataset_metrics_all,
                    model_metrics_all=validation_error_all.reset_index(drop=True),
                    model_type=model_type,
                    hyperparameters=best_params,
                    excel_filename=excel_filename,
                    param_summary=param_summary,
                )

                completed_models.append(model_type)

                gc.collect()

                print(f"Finished model: {model_type}")
                print(f"Runtime: {run_time}")

        sv.data_out = None
        sv.seq_reduction = None
        sv.data_input = None
        gc.collect()

        RUN_STATE["result"] = (
            f"Pipeline completed for: {', '.join(completed_models)}. "
            f"Results saved to {excel_filename}"
        )
        RUN_STATE["done"] = True
        RUN_STATE["running"] = False
        RUN_STATE["queue"].put(RUN_STATE["result"])

    except Exception as e:
        RUN_STATE["error"] = str(e)
        RUN_STATE["done"] = True
        RUN_STATE["running"] = False

        RUN_STATE["queue"].put("ERROR:")
        RUN_STATE["queue"].put(str(e))
        RUN_STATE["queue"].put(traceback.format_exc())

    finally:
        if optuna_logger is not None:
            try:
                optuna_logger.handlers = []
                optuna_logger.setLevel(old_optuna_level)
                optuna_logger.propagate = old_optuna_propagate

                for handler in old_optuna_handlers:
                    optuna_logger.addHandler(handler)
            except Exception:
                pass

def register_callbacks(app):
    config = load_config()

    @app.callback(
        Output("dataset-option-content", "children"),
        Input("dataset-option-dropdown", "value"),
    )
    def display_dataset_selection_feedback_callback(selected_dataset):
        logger.info(f"Dataset selected: {selected_dataset}")

        if selected_dataset is None:
            return html.Div("Please select a dataset.", className="subheader_1")

        return html.Div(f"Option '{selected_dataset}' selected.", className="subheader_1")

    @app.callback(
        [
            Output("dataset-dropdown", "options"),
            Output("dataset-dropdown-container", "style"),
            Output("dataset-selection-feedback", "children"),
        ],
        Input("dataset-option-dropdown", "value"),
        State("data-type-store", "data"),
    )
    def update_dataset_option_content(selected_option, data_type):
        labels = []
        options = []
        dropdown_style = {"display": "none"}
        info = html.Div(" ", className="subheader_1")

        if selected_option is not None:
            if data_type is None:
                data_type = "Wind"

            if selected_option == "new_dataset":
                try:
                    labels = [
                        f
                        for f in os.listdir(RAW_DATASETS_DIR)
                        if os.path.isdir(os.path.join(RAW_DATASETS_DIR, f))

                    ]
                except FileNotFoundError:
                    info = html.Div(
                        f"RAW_DATASETS_DIR '{RAW_DATASETS_DIR}' not found.",
                        className="subheader_1",
                    )

            elif selected_option == "select_from_folder":
                try:
                    labels = [
                        f
                        for f in os.listdir(INSERTED_DATASETS_DIR)
                        if os.path.isfile(os.path.join(INSERTED_DATASETS_DIR, f))
                    ]
                except FileNotFoundError:
                    info = html.Div(
                        f"INSERTED_DATASETS_DIR '{INSERTED_DATASETS_DIR}' not found.",
                        className="subheader_1",
                    )

            if labels:
                options = [{"label": name, "value": name} for name in labels]
                dropdown_style = {"display": "block"}
            else:
                info = html.Div("The data folder is empty", className="subheader_1")

        return options, dropdown_style, info

    @app.callback(
        Output("processed-data-store", "data"),
        [
            Input("dataset-dropdown", "value"),
            Input("confirm-categorical-choices", "n_clicks"),
        ],
        [
            State("dataset-option-dropdown", "value"),
            State({"type": "categorical-choice", "index": ALL}, "value"),
            State({"type": "categorical-choice", "index": ALL}, "id"),
            State("categorization-method-dropdown", "value"),
            State("processed-data-store", "data"),
        ],
    )
    def update_processed_data_store(
        selected_dataset,
        n_clicks_c,
        dataset_option,
        choices,
        ids,
        categorical_method,
        existing_data_store,
    ):
        ctx = callback_context

        if not ctx.triggered:
            logger.info("No trigger detected for the callback.")
            raise PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        logger.info(f"Callback triggered by: {triggered_id}")

        if triggered_id == "dataset-dropdown":
            if selected_dataset is None:
                return no_update

            if dataset_option == "new_dataset":
                dataset_path = RAW_DATASETS_DIR
            elif dataset_option == "select_from_folder":
                dataset_path = INSERTED_DATASETS_DIR
            else:
                return no_update

            dataset_f_path = os.path.join(dataset_path, selected_dataset)
            r_data, metadata = process_data_and_get_info(dataset_f_path)

            if r_data is not None:
                logger.info(f"Data and metadata loaded successfully from {dataset_path}.")
                return {
                    "data": r_data.to_json(date_format="iso", orient="split"),
                    "metadata": metadata,
                }

            logger.error(f"Failed to load data from {dataset_path}.")
            return no_update

        if triggered_id == "confirm-categorical-choices":
            if not (n_clicks_c > 0 and existing_data_store and existing_data_store.get("data")):
                logger.warning("No data available for categorical processing.")
                return no_update

            try:
                r_data_json = existing_data_store.get("data")
                r_data = pd.read_json(io.StringIO(r_data_json), orient="split")
                logger.info("Loaded data for categorical processing.")
            except Exception as e:
                logger.error(f"Error parsing existing_data_store: {e}")
                return no_update

            user_choices = {}
            for choice, id_dict in zip(choices, ids):
                column = id_dict["index"]
                user_choices[column] = choice

            logger.info(f"User choices for categorization: {user_choices}")

            try:
                r_data_processed, categorized_variables, excluded_variables = handle_categorical_columns(
                    r_data,
                    user_choices,
                    categorical_method,
                )
                logger.info("Categorical columns processed successfully.")
            except Exception as e:
                logger.error(f"Error processing categorical columns: {e}")
                return no_update

            metadata = existing_data_store.get("metadata", {}).copy()

            metadata.update(
                {
                    "categorized_variables": categorized_variables,
                    "excluded_variables": excluded_variables,
                    "categorization_method": categorical_method,
                }
            )

            return {
                "data": r_data_processed.to_json(date_format="iso", orient="split"),
                "metadata": metadata,
            }

        logger.warning(f"Unhandled trigger: {triggered_id}")
        return no_update

    @app.callback(
        [
            Output("categorical-options-container", "children"),
            Output("categorization-method-dropdown", "options"),
            Output("categorization-method-dropdown", "value"),
            Output("categorization-method-container", "style"),
            Output("graphs-container_t", "style"),
        ],
        Input("processed-data-store", "data"),
    )
    def display_categorical_options_callback(processed_data_store):
        logger.info("Displaying categorical options.")

        if processed_data_store is None or not processed_data_store.get("data"):
            return None, [], None, {"display": "none"}, {"display": "none"}

        metadata = processed_data_store.get("metadata", {})

        if metadata and has_valid_categorization(metadata):
            categorization_methods = [
                {"label": method.capitalize().replace("_", " "), "value": method}
                for method in config["categorical_methods"]
            ]

            return (
                html.Div(" ", className="subheader_1"),
                categorization_methods,
                None,
                {"display": "none"},
                {"display": "block"},
            )

        try:
            r_data_json = processed_data_store.get("data")
            r_data = pd.read_json(io.StringIO(r_data_json), orient="split")
        except Exception as e:
            logger.error(f"Error parsing processed_data_store: {e}")
            return (
                html.Div("Error processing data.", className="subheader_1"),
                [],
                None,
                {"display": "none"},
                {"display": "none"},
            )

        selected_columns = r_data.columns[
            (r_data.dtypes == "object") | (r_data.isna().all())
        ].tolist()

        logger.info(f"Object or NAN columns found: {selected_columns}")

        if selected_columns:
            header_row = dbc.Row(
                [
                    dbc.Col(html.B("Column"), width=6),
                    dbc.Col(
                        html.B("Categorize / Exclude"),
                        width=6,
                        style={"text-align": "center", "padding-left": "20px"},
                    ),
                ],
                className="table-header",
            )

            categorical_options = [header_row]

            for col in selected_columns:
                categorical_options.append(
                    dbc.Row(
                        [
                            dbc.Col(html.Label(f" {col}"), width=6),
                            dbc.Col(
                                dcc.RadioItems(
                                    id={"type": "categorical-choice", "index": col},
                                    options=[
                                        {"label": "", "value": "categorize"},
                                        {"label": "", "value": "exclude"},
                                    ],
                                    value="categorize",
                                    inline=True,
                                    className="custom-radio_t",
                                    labelStyle={
                                        "display": "inline-block",
                                        "margin-right": "30px",
                                        "margin-left": "30px",
                                    },
                                ),
                                width=6,
                            ),
                        ],
                        className="table-row",
                    )
                )

            categorization_methods = [
                {"label": method.capitalize().replace("_", " "), "value": method}
                for method in config["categorical_methods"]
            ]

            return (
                html.Div(
                    [
                        html.H4("Handle Categorical Columns:", className="header_1"),
                        html.Div(categorical_options),
                    ],
                    className="categorical-options-section",
                ),
                categorization_methods,
                categorization_methods[0]["value"] if categorization_methods else None,
                {"display": "block"},
                {"display": "none"},
            )

        return (
            html.Div(" ", className="subheader_1"),
            [],
            None,
            {"display": "none"},
            {"display": "block"},
        )

    @app.callback(
        [
            Output("dataframe-info", "children"),
            Output("dataframe-head", "children"),
            Output("metadata-display", "children"),
            Output("dataframe-info", "style"),
            Output("dataframe-head", "style"),
        ],
        Input("processed-data-store", "data"),
    )
    def update_dataframe_display(processed_data_store):
        if not processed_data_store or not processed_data_store.get("data"):
            return None, None, None, {"display": "none"}, {"display": "none"}

        try:
            data_json = processed_data_store.get("data")
            metadata = processed_data_store.get("metadata", {})
            data = pd.read_json(io.StringIO(data_json), orient="split")
        except Exception as e:
            logger.error(f"Error parsing processed_data_store: {e}")
            return None, None, None, {"display": "none"}, {"display": "none"}

        if metadata and has_valid_categorization(metadata):
            suffix = " (After Processing)"

            categorization_method = metadata.get("categorization_method", "Not Specified")
            categorized_variables = metadata.get("categorized_variables", [])
            excluded_variables = metadata.get("excluded_variables", [])

            categorized_vars_str = ", ".join(categorized_variables) if categorized_variables else "None"
            excluded_vars_str = ", ".join(excluded_variables) if excluded_variables else "None"

            metadata_display = html.Div(
                [
                    html.H5("Categorization Metadata:", className="header_1"),
                    html.P(f"Categorization Method: {categorization_method}"),
                    html.P(f"Categorized Variables: {categorized_vars_str}"),
                    html.P(f"Excluded Variables: {excluded_vars_str}"),
                ],
                className="metadata-display",
            )

        else:
            suffix = " "
            metadata_display = html.Div(
                "No categorization metadata available.",
                className="subheader_1",
            )

        dataframe_info = html.Div(
            [
                html.H5(f"Dataframe Info{suffix}:", className="header_1"),
                html.Pre(get_dataframe_info(data)),
            ],
            className="dataframe-info",
        )

        dataframe_head = generate_html_table(data, suffix)

        return dataframe_info, dataframe_head, metadata_display, {"display": "block"}, {"display": "block"}

    @app.callback(
        [
            Output("input-features-checkbox", "options"),
            Output("input-features-checkbox", "value"),
            Output("output-variable-radio", "options"),
            Output("output-variable-radio", "value"),
            Output("input-output-variables-container", "style"),
        ],
        [
            Input("processed-data-store", "data"),
            Input("select-all-features", "value"),
        ],
    )
    def display_input_output_variables_callback(processed_data_store, select_all_features):
        if not processed_data_store or not processed_data_store.get("data"):
            return [], [], [], None, {"display": "none"}

        try:
            data = pd.read_json(
                io.StringIO(processed_data_store.get("data")),
                orient="split",
            )

            metadata = processed_data_store.get("metadata", {})
            columns = data.columns.tolist()

            input_features_options = [
                {"label": col, "value": col}
                for col in columns
            ]

            output_variable_options = [
                {"label": col, "value": col}
                for col in columns
            ]

            output_variable = metadata.get("output_variable", None)

            ctx = callback_context
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

            if triggered_id == "select-all-features":
                if select_all_features and "ALL" in select_all_features:
                    input_features = columns.copy()
                else:
                    input_features = []
            else:
                input_features = metadata.get("input_features", [])
                if input_features is None:
                    input_features = []

            return (
                input_features_options,
                input_features,
                output_variable_options,
                output_variable,
                {"display": "block"},
            )

        except Exception as e:
            logger.error(f"Error in display_input_output_variables_callback: {e}")
            return [], [], [], None, {"display": "none"}

    @app.callback(
        [
            Output("models-checkbox", "options"),
            Output("models-checkbox", "value"),
            Output("feature-filtering-radio", "options"),
            Output("feature-filtering-radio", "value"),
        ],
        [
            Input("processed-data-store", "data"),
            Input("select-all-models", "value"),
        ],
    )
    def populate_pipeline_choices(processed_data_store, select_all_models):
        active_config = load_config()

        model_values = active_config["ml_model"] + active_config["deep_model"]
        filtering_values = active_config["feature_filtering"]

        model_options = [
            {"label": model, "value": model}
            for model in model_values
        ]

        filtering_options = [
            {"label": method, "value": method}
            for method in filtering_values
        ]

        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if triggered_id == "select-all-models":
            if select_all_models and "ALL" in select_all_models:
                selected_models = model_values
            else:
                selected_models = []

            return model_options, selected_models, filtering_options, no_update

        selected_models = []
        selected_filtering = None

        if processed_data_store and processed_data_store.get("metadata"):
            metadata = processed_data_store.get("metadata", {})

            selected_models = metadata.get("model", [])
            selected_filtering = metadata.get("feature_filtering")

            if selected_models is None:
                selected_models = []

            if isinstance(selected_models, str):
                selected_models = [selected_models]

        return model_options, selected_models, filtering_options, selected_filtering

    @app.callback(
        Output("file-name-input", "value"),
        Input("dataset-dropdown", "value"),
    )
    def suggest_file_name(dataset_name):
        if dataset_name:
            return f"{dataset_name}_processed"

        return ""

    @app.callback(
        Output("save-container_t", "style"),
        Input("processed-data-store", "data"),
    )
    def toggle_save_container(processed_data_store):
        if processed_data_store and processed_data_store.get("data"):
            return {"display": "block"}

        return {"display": "none"}

    @app.callback(
        Output("save-feedback", "children"),
        Input("save-data-button", "n_clicks"),
        State("processed-data-store", "data"),
        State("file-name-input", "value"),
        State("input-features-checkbox", "value"),
        State("output-variable-radio", "value"),
        State("feature-filtering-radio", "value"),
        State("models-checkbox", "value"),
        prevent_initial_call=True,
    )
    def save_processed_data(
        n_clicks,
        processed_data_store,
        file_name,
        selected_features,
        selected_output,
        selected_feature_filtering,
        selected_models,
    ):
        if n_clicks <= 0:
            return dbc.Alert("Click 'Save Data' to save the processed data.", color="info")

        if not (
            processed_data_store
            and processed_data_store.get("data")
        ):
            logger.warning("No processed data available to save.")
            return dbc.Alert("No processed data available to save.", color="warning")

        if not file_name or file_name.strip() == "":
            logger.warning("Invalid file name provided.")
            return dbc.Alert("Please provide a valid file name.", color="warning")

        if not selected_features:
            return dbc.Alert("No input features selected.", color="warning")

        if not selected_output:
            return dbc.Alert("No output variable selected.", color="warning")

        if not selected_feature_filtering:
            return dbc.Alert("No feature filtering method selected.", color="warning")

        if not selected_models:
            return dbc.Alert("No model selected.", color="warning")

        sanitized_file_name = sanitize_filename(file_name)
        logger.info(f"Sanitized file name: '{sanitized_file_name}'")

        try:
            processed_data = pd.read_json(
                io.StringIO(processed_data_store.get("data")),
                orient="split",
            )
            logger.info(f"Processed data loaded successfully with shape {processed_data.shape}")
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return dbc.Alert(f"Error loading processed data: {e}", color="danger")

        metadata = processed_data_store.get("metadata", {}).copy()

        metadata_to_save = {
            "categorized_variables": metadata.get("categorized_variables", []),
            "excluded_variables": metadata.get("excluded_variables", []),
            "categorization_method": metadata.get("categorization_method", None),
            "input_features": selected_features,
            "output_variable": selected_output,
            "features_creation": metadata.get("features_creation", None),
            "feature_filtering": selected_feature_filtering,
            "classification_method": metadata.get("classification_method", None),
            "model": selected_models,
        }

        save_path = os.path.join(INSERTED_DATASETS_DIR, f"{sanitized_file_name}.pkl")
        logger.info(f"Preparing to save data to {save_path}")

        if not os.path.exists(INSERTED_DATASETS_DIR):
            try:
                os.makedirs(INSERTED_DATASETS_DIR)
                logger.info(f"Created directory '{INSERTED_DATASETS_DIR}'")
            except Exception as e:
                logger.error(f"Error creating directory '{INSERTED_DATASETS_DIR}': {e}")
                return dbc.Alert(
                    f"Error creating directory '{INSERTED_DATASETS_DIR}': {e}",
                    color="danger",
                )

        if os.path.exists(save_path):
            logger.warning(f"File '{save_path}' already exists.")
            return dbc.Alert(
                f"File '{save_path}' already exists. Please choose a different name.",
                color="warning",
            )

        try:
            with open(save_path, "wb") as f:
                save_data = {
                    "data": processed_data,
                    "metadata": metadata_to_save,
                }
                pd.to_pickle(save_data, f)

            logger.info(f"Data successfully saved to {save_path}")
            return dbc.Alert(f"Data successfully saved to {save_path}", color="success")

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return dbc.Alert(f"Error saving data: {e}", color="danger")

    @app.callback(
        Output("run-pipeline-container", "style"),
        Input("processed-data-store", "data"),
    )
    def toggle_run_pipeline_visibility(processed_data_store):
        if processed_data_store and processed_data_store.get("data"):
            return {"display": "block"}

        return {"display": "none"}

    @app.callback(
        [
            Output("run-log-interval", "disabled"),
            Output("run-pipeline-output", "children"),
        ],
        Input("run-pipeline-button", "n_clicks"),
        State("processed-data-store", "data"),
        State("dataset-dropdown", "value"),
        State("run-excel-file-name", "value"),
        State("input-features-checkbox", "value"),
        State("output-variable-radio", "value"),
        State("feature-filtering-radio", "value"),
        State("models-checkbox", "value"),
        prevent_initial_call=True,
    )
    def start_pipeline_run(
        n_clicks,
        processed_data_store,
        dataset_name,
        excel_file_name,
        selected_features,
        selected_output,
        selected_feature_filtering,
        selected_models,
    ):
        if n_clicks <= 0:
            raise PreventUpdate

        if RUN_STATE["running"]:
            return False, dbc.Alert("Pipeline is already running.", color="warning")

        if not (
            processed_data_store
            and processed_data_store.get("data")
        ):
            return True, dbc.Alert(
                "No processed data available to run the pipeline.",
                color="warning",
            )

        metadata = processed_data_store.get("metadata", {}).copy()

        metadata["input_features"] = selected_features or []
        metadata["output_variable"] = selected_output
        metadata["feature_filtering"] = selected_feature_filtering
        metadata["model"] = selected_models or []

        if not metadata["input_features"]:
            return True, dbc.Alert("No input features selected.", color="warning")

        if not metadata["output_variable"]:
            return True, dbc.Alert("No output variable selected.", color="warning")

        if not metadata["feature_filtering"]:
            return True, dbc.Alert("No feature filtering method selected.", color="warning")

        if not metadata["model"]:
            return True, dbc.Alert("No model selected.", color="warning")

        reset_run_state()

        RUN_STATE["running"] = True
        RUN_STATE["done"] = False

        worker = threading.Thread(
            target=run_models_worker,
            args=(
                processed_data_store.get("data"),
                metadata,
                dataset_name,
                excel_file_name,
            ),
            daemon=True,
        )

        RUN_STATE["thread"] = worker
        worker.start()

        return False, dbc.Alert("Pipeline started.", color="info")

    @app.callback(
        [
            Output("run-terminal-output", "children"),
            Output("run-log-interval", "disabled", allow_duplicate=True),
            Output("run-pipeline-output", "children", allow_duplicate=True),
        ],
        Input("run-log-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def update_run_terminal(_):
        logs = drain_run_logs()

        if RUN_STATE["done"]:
            if RUN_STATE["error"]:
                return (
                    logs,
                    True,
                    dbc.Alert(f"Pipeline failed: {RUN_STATE['error']}", color="danger"),
                )

            return (
                logs,
                True,
                dbc.Alert(RUN_STATE["result"], color="success"),
            )

        return logs, False, no_update

    app.clientside_callback(
        """
        function(log_text) {
            const el = document.getElementById("run-terminal-output");
            if (!el) {
                return window.dash_clientside.no_update;
            }

            const threshold = 40;
            const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
            const wasNearBottom = distanceFromBottom < threshold;

            setTimeout(function() {
                const currentDistanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;

                if (wasNearBottom || currentDistanceFromBottom < threshold) {
                    el.scrollTop = el.scrollHeight;
                }
            }, 0);

            return window.dash_clientside.no_update;
        }
        """,
        Output("run-terminal-wrapper", "data-scroll-dummy"),
        Input("run-terminal-output", "children"),
    )

    display_graph.register_callbacks(app)