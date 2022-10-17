import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, ctx, State
import base64
import io
import time
import os
from dash.exceptions import PreventUpdate
from fit3162_fit3164.custom.code.audio2lm.video_generation import video_generation

# external_stylesheets = ["assets/fontawesome-free-5.15.4-web/css/fontawesome.min.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "margin-top": "2.3%",
    "border-image": "linear-gradient(#B58ECC,#5DE6DE)",
    "border": "5px solid"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "5rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
    "width": "100%",
    "border-image": "linear-gradient(#B58ECC,#5DE6DE)",
    "border": "5px solid",
    "background-color": "#222831"

}

sidebar = html.Div(
    [

        html.P(
            "It provide the user with a platform to drop two audio files, and then at the back end, respective methods are called to generate the video which is then displayed on the user’s computer screens",
            className="lead",
            style={"font-size": "90%", "color": "black"}
        ),
        html.Br(),
        dcc.Upload(html.Button("Upload Content Audio File", role="button", className="button-54"), id="upload-data-1"),
        html.Hr(),
        dcc.Upload(html.Button("Upload Emotional Audio File", role="button", className="button-54"),
                   id="upload-data-2"),

        html.Div(children=[html.Div(id="output-1"), html.Div(id="output-2")],
                 style={"margin-top": "5rem", "margin-bottom": "5rem", "height": "5%"}),
        dbc.Row([dbc.Col(

            html.Button("Render Video", role="button", className="button-36", id="btn-nclicks-1", n_clicks=0), width=6,
            style={"padding-left": "0"}
        ), dbc.Col(
            html.Button("Launch App", role="button", className="button-36", id="btn-nclicks-2", n_clicks=0), width=6,
            style={"padding-left": "0"}
        )])

    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(dcc.Loading(id="Loading1", children=html.Div(children=html.B("Please Press the Launch button first", style={"color": "white"}),id="page-content"), type="circle"), style=CONTENT_STYLE)
# content = html.Div(children=[html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",className="img1")],id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dbc.Row(html.Div("Facial Landmarks App", className="head"), style={"height": "120%"}),
                       dbc.Row([
                           dbc.Col(sidebar, width=3),
                           dbc.Col(content, width=9),
                       ])])


@app.callback(Output("output-1", "children"), [Input('upload-data-1', 'contents'),Input('btn-nclicks-2', 'n_clicks')], State('upload-data-1', 'filename'))
def upload_data(content,btn2, filename):
    if "btn-nclicks-2" == ctx.triggered_id:
        return ""
    elif content is not None:
        if save_audio_file(content, filename, "1"):
            return "successful upload of content audio file"
        else:
            return "Problem occured during upload.ReUpload"


@app.callback(Output("output-2", "children"), [Input('upload-data-2', 'contents'),Input('btn-nclicks-2', 'n_clicks')], State('upload-data-2', 'filename'))
def upload_data_2(content,btn2,filename):
    if "btn-nclicks-2" == ctx.triggered_id:
        return ""
    elif content is not None:
        if save_audio_file(content, filename, "2"):
            return "successful upload of emotional audio file"
        else:
            return "Problem occured during upload.ReUpload"


def save_audio_file(content, filename, num):
    if "wav" in filename:

        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        fh = open("file" + num + ".wav", "wb")
        fh.write(decoded)
        fh.close()
        return True
    else:
        return False


@app.callback(
    Output("page-content", "children"),
    [Input('btn-nclicks-1', 'n_clicks'),
     Input('btn-nclicks-2', 'n_clicks')]
)
def button_Click(btn1, btn2):
    if "btn-nclicks-1" == ctx.triggered_id and btn2>0:
            if os.path.exists("file1.wav") and os.path.exists("file2.wav"):
                video_generation(["file1.wav", "file2.wav"])
                return html.Nav(html.Video(
                    controls=True,
                    src='assets/video_test_with_audio.mp4',
                    height=600,
                    width= 700,

                ))
            else:
                return html.B("Audio files are not provided", style={"color": "white"})


    elif "btn-nclicks-2" == ctx.triggered_id:
         if os.path.exists("file1.wav"):
            os.remove("file1.wav")
         if os.path.exists("file2.wav"):
            os.remove("file2.wav")
         if os.path.exists("assets/video_test_with_audio.mp4"):
             os.remove("assets/video_test_with_audio.mp4")
         return html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",
                        className="img1")
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True, port=9999, dev_tools_hot_reload=False)