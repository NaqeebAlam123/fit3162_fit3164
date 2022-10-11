# from dash import Dash, dcc
# import dash_bootstrap_components as dbc

# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# mytext = dcc.Markdown(children="# Hello World - let's build web apps in Python")

# app.layout = dbc.Container([mytext])

# if __name__ == '__main__':
#     app.run_server(port=8051)



# from dash import Dash, html


# app = Dash(__name__)

# app.layout = html.Nav(html.Video(
#             controls = True,
#             src = "D:\Project2\video\video\front\angry\level_1\005.mp4",
#         ))
    

# if __name__ == '__main__':
#     app.run_server(debug=True, use_reloader=False)




# import dash
# from dash import html

# app = dash.Dash(__name__)

# app.layout = html.Nav(html.Video(
#              controls = True,
#              src = "assets/005.mp4",
#              height=600,
#          ))

# if __name__ == '__main__':
#     app.run_server(debug=True)




# from dash import Dash, dcc, html, Input, Output

# app = Dash(__name__)
# app.layout = html.Div([
#    html.Label(),
#    # dcc.Dropdown(['ANGRY', 'SAD', 'NEUTRAL'], 'NEUTRAL', id='demo-dropdown'),
#     html.Div(id='dd-output-container')
# ])


# @app.callback(
#     Output('dd-output-container', 'children'),
#     Input('demo-dropdown', 'value')
# )


# def update_output(value):
#     if value=='SAD':
#         result=html.Nav(html.Video(
#              controls = True,
#              src = "assets/video_happy_with_audio(1).mp4",
#              #height=600,
#          ))
#     elif value=='NEUTRAL':
#         result=html.Nav(html.Video(
#              controls = True,
#              src = "assets/005.mp4",
#              height=600,
#          ))
#     elif value=='ANGRY':
#         result=html.Nav(html.Video(
#              controls = True,
#              src = "assets/014.mp4",
#              height=600,
#          ))    
#     return result



# if __name__ == '__main__':
#     app.run_server(debug=True)



# import base64
# from dash import Dash, dcc, html, Input, Output

# app = Dash(__name__)
# app.layout =app.layout = html.Div([
#     dcc.Upload(
#         id='upload-data',
#         children=html.Div([
#             'Drag and Drop or ',
#             html.A('Select Files')
#         ]),
#         style={
#             'width': '100%',
#             'height': '60px',
#             'lineHeight': '60px',
#             'borderWidth': '1px',
#             'borderStyle': 'dashed',
#             'borderRadius': '5px',
#             'textAlign': 'center',
#             'margin': '10px'
#         },
#         # Allow multiple files to be uploaded
#         multiple=True
#     ),
#     html.Div(id='output-data-upload', style={
#             'width': '100%',
#             'height': '60px',
#             'textAlign': 'center',
#             'margin': '10px'
#         }),
# ])


# @app.callback(
#     Output('output-data-upload', 'children'),
#     Input('upload-data', 'filename')
# )


# def update_output(filename):
#     results=[]
#     for file in filename:
#         results.append(file.split('_')[0])
#         #print(file)
    
#     return results
    
#     if results=="angry":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_angry_with_audio.mp4",
#              height=600,
#           ))
#     elif results=="contempt":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_angry_with_audio.mp4",
#              height=600,
#           ))
#     elif results=="disgusted":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_disgusted_with_audio.mp4",
#              height=600,
#           ))
#     elif results=="fear":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_fear_with_audio.mp4",
#              height=600,
#           ))
#     elif results=="neutral":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_neutral_with_audio.mp4",
#              height=600,
#           ))
#     elif results=="sad":
#         output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_sad_with_audio.mp4",
#              height=600,
#           ))

#     elif results=="surprised":
#          output=html.Nav(html.Video(
#               controls = True,
#               src = "assets/video_surprised_with_audio.mp4",
#              height=600,
#           ))

#     else:
#         output= "Not found"    
   
#     return output



# if __name__ == '__main__':
#     app.run_server(debug=True)




# import dash
# import dash_bootstrap_components as dbc
# from dash import Input, Output, dcc, html,ctx,State
# import base64
# import io
# import time
# import os
# from fit3162_fit3164.custom.code.audio2lm.video_generation import video_generation


# #external_stylesheets = ["assets/fontawesome-free-5.15.4-web/css/fontawesome.min.css"]

# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

# # the style arguments for the sidebar. We use position:fixed and a fixed width
# SIDEBAR_STYLE = {
#     "position": "fixed",
#     "top": 0,
#     "left": 0,
#     "bottom": 0,
#     "width": "20rem",
#     "padding": "2rem 1rem",
#     "background-color": "#f8f9fa",
#     "margin-top":"2.3%",
#     "border-image":"linear-gradient(#B58ECC,#5DE6DE)",
#     "border": "5px solid"
# }

# # the styles for the main content position it to the right of the sidebar and
# # add some padding.
# CONTENT_STYLE = {
#     "margin-left": "5rem",
#     "margin-right": "1rem",
#     "padding": "2rem 1rem",
#     "width":"100%",
#     "border-image":"linear-gradient(#B58ECC,#5DE6DE)",
#     "border": "5px solid",
#     "background-color": "#222831"

# }

# sidebar = html.Div(
#     [

#         html.P(
#             "It provide the user with a platform to drop two audio files, and then at the back end, respective methods are called to generate the video which is then displayed on the user’s computer screens", className="lead",
#             style={"font-size":"90%","color":"black"}
#         ),
#         html.Br(),
#         dcc.Upload(html.Button("Upload Content Audio File", role="button", className="button-54"),id="upload-data-1"),
#         html.Hr(),
#         dcc.Upload(html.Button("Upload Emotional Audio File", role="button", className="button-54"),id="upload-data-2"),


#         html.Div(children=[html.Div(id="output-1"),html.Div(id="output-2")],style={"margin-top":"5rem","margin-bottom":"5rem","height":"5%"}),
#         dbc.Row([dbc.Col(

#        html.Button("Render Video", role="button", className="button-36",id="btn-nclicks-1",n_clicks=0),width=6,style={"padding-left":"0"}
#         ),dbc.Col(
#        html.Button("Reset App", role="button", className="button-36",id="btn-nclicks-2",n_clicks=0),width=6,style={"padding-left":"0"}
#         )])

#     ],
#     style=SIDEBAR_STYLE,
# )
# content=html.Div(dcc.Loading(id="Loading1",children=html.Div(id="page-content"),type="circle"),style=CONTENT_STYLE)
# #content = html.Div(children=[html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",className="img1")],id="page-content", style=CONTENT_STYLE)

# app.layout = html.Div( [dbc.Row(html.Div("Facial Landmarks App",className="head"),style={"height":"120%"}),
#                        dbc.Row([
#                            dbc.Col(sidebar,width=3),
#                            dbc.Col(content,width=9),
#                        ])])

# @app.callback(Output("output-1","children"),Input('upload-data-1', 'contents'),State('upload-data-1', 'filename'))
# def upload_data(content,filename):
#     if content is not None:
#         if save_audio_file(content,filename,"1"):
#             return "successful upload of content audio file"
#         else:
#             return "Problem occured during upload.ReUpload"



# @app.callback(Output("output-2","children"),Input('upload-data-2', 'contents'),State('upload-data-2', 'filename'))
# def upload_data_2(content,filename):
#         if content is not None:
#             if save_audio_file(content, filename,"2"):
#                 return "successful upload of emotional audio file"
#             else:
#                 return "Problem occured during upload.ReUpload"

# def save_audio_file(content,filename,num):
#    if "wav" in filename:

#      content_type, content_string = content.split(',')
#      decoded = base64.b64decode(content_string)
#      fh = open("file"+num+".wav", "wb")
#      fh.write(decoded)
#      fh.close()
#      return True
#    else:
#      return False


# @app.callback(
#     Output("page-content", "children"),
#     [Input('btn-nclicks-1', 'n_clicks'),
#     Input('btn-nclicks-2', 'n_clicks'),
#     Input("page-content","children")]
# )
# def button_Click(btn1, btn2, default):
#     if "btn-nclicks-1" == ctx.triggered_id:
#         if not os.path.exists("assets/video_test_with_audio.mp4"):
#          if os.path.exists("file1.wav") and os.path.exists("file2.wav"):
#            video_generation(["file1.wav", "file2.wav"])
#          else:
#            return html.Div("Audio files are not provided",style={"color":"white"}) 
        
#         time.sleep(6)
#         return html.Nav(html.Video(
#                 controls = True,
#                 src ="assets/video_test_with_audio.mp4",
#                 height=600
#             ))
#     elif "btn-nclicks-2" == ctx.triggered_id:
#         if os.path.exists("assets/video_test_with_audio.mp4"):
#             os.remove("assets/video_test_with_audio.mp4")
#         if os.path.exists("file1.wav"):
#             os.remove("file1.wav")
#         if os.path.exists("file2.wav"):
#             os.remove("file2.wav")
#         return html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",className="img1")
    


# # @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
# # def render_page_content(pathname):
# #     if pathname == "/":
# #         return html.Div(html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",className="img1"))
# #     elif pathname == "/page-1":
# #         return html.P("This is the content of page 1. Yay!")
# #     elif pathname == "/page-2":
# #         return html.P("Oh cool, this is page 2!")
# #     # If the user tries to reach a different page, return a 404 message
# #     return html.Div(
# #         [
# #             html.H1("404: Not found", className="text-danger"),
# #             html.Hr(),
# #             html.P("The pathname" +pathname+" was not recognised..."),
# #         ],
# #         className="p-3 bg-light rounded-3",
# #     )




# if __name__ == "__main__":
#     app.run_server(debug=True,port=9999)




import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html,ctx,State
from fit3162_fit3164.custom.code.audio2lm.video_generation import video_generation
import base64
import io
import time
import os

#external_stylesheets = ["assets/fontawesome-free-5.15.4-web/css/fontawesome.min.css"]

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "margin-top":"2.3%",
    "border-image":"linear-gradient(#B58ECC,#5DE6DE)",
    "border": "5px solid"
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "5rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
    "width":"100%",
    "border-image":"linear-gradient(#B58ECC,#5DE6DE)",
    "border": "5px solid",
    "background-color": "#222831"

}

sidebar = html.Div(
    [

        html.P(
            "It provide the user with a platform to drop two audio files, and then at the back end, respective methods are called to generate the video which is then displayed on the user’s computer screens", className="lead",
            style={"font-size":"90%","color":"black"}
        ),
        html.Br(),
        dcc.Upload(html.Button("Upload Neutral Audio File", role="button", className="button-54"),id="upload-data-1"),
        html.Hr(),
        dcc.Upload(html.Button("Upload Emotional Audio File", role="button", className="button-54"),id="upload-data-2"),


        html.Div(children=[html.Div(id="output-1"),html.Div(id="output-2")],style={"margin-top":"5rem","margin-bottom":"5rem","height":"5%"}),
        dbc.Row([dbc.Col(

       html.Button("Render Video", role="button", className="button-36",id="btn-nclicks-1",n_clicks=0),width=6,style={"padding-left":"0"}
        ),dbc.Col(
       html.Button("Launch", role="button", className="button-36",id="btn-nclicks-2",n_clicks=0),width=6,style={"padding-left":"0"}
        )])

    ],
    style=SIDEBAR_STYLE,
)
content=html.Div(dcc.Loading(id="Loading1",children=html.Div(id="page-content"),type="circle"),style=CONTENT_STYLE)
#content = html.Div(children=[html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",className="img1")],id="page-content", style=CONTENT_STYLE)

app.layout = html.Div( [dcc.Store(id='button-previous-1',data=0),
                        dcc.Store(id="button-previous-2",data=0),
                       dbc.Row(html.Div("Facial Landmark Generation App",className="head"),style={"height":"120%"}),
                       dbc.Row([
                           dbc.Col(sidebar,width=3),
                           dbc.Col(content,width=9),
                       ])])

@app.callback(Output("output-1","children"),Input('upload-data-1', 'contents'),State('upload-data-1', 'filename'))
def upload_data(content,filename):
    if content is not None:
        if save_audio_file(content,filename,"1"):
            return "successful upload of neutral audio file"
        else:
            return "Problem occured during upload.ReUpload"



@app.callback(Output("output-2","children"),Input('upload-data-2', 'contents'),State('upload-data-2', 'filename'))
def upload_data_2(content,filename):
        if content is not None:
            if save_audio_file(content, filename,"2"):
                return "successful upload of emotional audio file"
            else:
                return "Problem occured during upload.ReUpload"

def save_audio_file(content,filename,num):
   if "wav" in filename:

     content_type, content_string = content.split(',')

     decoded = base64.b64decode(content_string)
     fh = open("file"+num+".wav", "wb")
     fh.write(decoded)
     fh.close()
     return True
   else:
     return False



list_of_inputs=[Input('btn-nclicks-1', 'n_clicks'),Input("button-previous-1","data"),Input('btn-nclicks-2', 'n_clicks'),Input("button-previous-2","data")]
list_of_outputs=[Output("page-content", "children"),Output("button-previous-1","data"),Output("button-previous-2","data")]
@app.callback(list_of_outputs,list_of_inputs)
def render_output(btn1,previous1,btn2,previous2):
    if btn1 > previous1 and btn2 >0:
        if os.path.exists("file1.wav") and os.path.exists("file2.wav"):
            video_generation(["file1.wav", "file2.wav"])
            return html.Nav(html.Video(
                controls=True,
                src="assets/video_test_with_audio.mp4",
                height=600
            )), btn1,previous2
        else:
            return html.B("Audio files not detected : Add files first",style={"color":"white"}),btn1,previous2
    elif btn2>previous2:
        if os.path.exists("file1.wav"):
          os.remove("file1.wav")
        elif os.path.exists("file2.wav"):
          os.remove("file2.wav")
        return html.Img(src="assets/Your paragraph text.png", alt="Grapefruit slice atop a pile of other slices",
                        className="img1"), previous1,btn2
    else:
        return html.B("Press the Launch button to pull up a loading screen",style={"color":"white"}),btn1,btn2




if __name__ == "__main__":
    app.run_server(debug=True,port=9999)