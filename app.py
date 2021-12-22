"""
Dashboard app running a Linear Regression model
to estimate the demand of bike rentals for Bike Share in Washington D.C.
"""

import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np 

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



''' Ridge model to predirct Bike count'''

pwd = os.getcwd()

df = pd.read_csv(pwd + '/train.csv', sep=",")


#prepare data
#create features
df['datetime'] = pd.to_datetime(df['datetime'])

def create_dt_features(df):
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['hour'] = pd.DatetimeIndex(df['datetime']).hour
    return df
    
    
df = create_dt_features(df)
df.set_index(df['datetime'],inplace=True)
df.drop('datetime', axis=1,inplace=True)

#pick features
y = df["count"]
X = df[['workingday', 'weather', 'temp', 'month', 'hour']]

#train a model
model = make_pipeline(PolynomialFeatures(degree=2),Ridge())

model.fit(X,y)


col_names = ['workingday', 'weather', 'temp', 'month', 'hour']
df_feature_importances = pd.DataFrame([6, 6, 8, 9, 10], columns=["Importance"],index=col_names)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
# Feature Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Feature Importance of the model<b>', title_x=0.5)


# get name, min, mean and max of the features
slider_1_label = "Temperature"
slider_1_min = 0
slider_1_mean = 20
slider_1_max = 45

slider_2_label = "Hour"
slider_2_min = 0
slider_2_mean = 11
slider_2_max = 23

slider_3_label = "Weather"
slider_3_min = 1
slider_3_mean = 1
slider_3_max = 4

slider_4_label = "Workingday"
slider_4_min = 0
slider_4_mean = 0
slider_4_max = 1

slider_5_label = "Month"
slider_5_min = 1
slider_5_mean = 6
slider_5_max = 12

options_months = [
    {'label': 'January', 'value': 1},
    {'label': 'February', 'value': 2},
    {'label': 'March', 'value': 3},
    {'label': 'April', 'value': 4},
    {'label': 'May', 'value': 5},
    {'label': 'June', 'value': 6},
    {'label': 'July', 'value': 7},
    {'label': 'August', 'value': 8},
    {'label': 'September', 'value': 9},
    {'label': 'October', 'value': 10},
    {'label': 'November', 'value': 11},
    {'label': 'December', 'value': 12}]


###############################################################################

app = dash.Dash()


# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    .
#    .
#    .
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output


# HTML formatting to the layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Arial','color':'#455a64'},
                      
                    children=[

                        html.H1(children="Bike Rentals Simulation Dashboard"),
                        
                        #Dash Graph Component calls the fig_features_importance parameters
                        dcc.Graph(figure=fig_features_importance),

                        # Dash Slider built according to Feature #1 ranges
                        #temperature
                        html.H4(children=slider_1_label),
                        
                        dcc.Slider(
                            id='X1_slider',
                            min=slider_1_min,
                            max=slider_1_max,
                            step=1.0,
                            value=slider_1_mean,
                            marks={i: '{}°'.format(i) for i in range(slider_1_min, slider_1_max+1,5)}
                            ),
                        
                        # The same logic is applied to the following names / sliders
                        #hour
                        html.H4(children=slider_2_label),
                        
                        dcc.Slider(
                            id='X2_slider',
                            min=slider_2_min,
                            max=slider_2_max,
                            step=1.0,
                            value=slider_2_mean,
                            marks={i: '{}h'.format(i) for i in range(slider_2_min, slider_2_max+1)}
                            ),
                        
                        #weather
                        html.H4(children=slider_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=slider_3_min,
                            max=slider_3_max,
                            step=1.0,
                            value=slider_3_mean,
                            marks={1: 'Clear', 2: 'Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'},
                            ),
                    
                        #workingday
                        html.H4(children=slider_4_label),
                        dcc.RadioItems(
                            id='X4_slider',
                            options=[{"label": "Yes", 'value':1},
                            {"label": "No", 'value':0}],
                            value=1
                            ),
                        
                        #month
                        html.H4(children=slider_5_label),

                        dcc.Dropdown(
                            id='X5_slider',
                            className="input-line",
                            style={"flex-grow":"3",},
                            options=options_months,
                            value=1
                            ),
                        
                        
                        # Prediction result
                        html.H2(id="prediction_result"),
                        
                    ])
                   

############################################################################################

# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value"),
               Input("X4_slider","value"), Input("X5_slider","value")])  #!!!

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3, X4, X5):

    # We create a NumPy array in the form of the original features
    # ["Temperature","Hour","Weather", "Workingday","Month"]
    input_X = np.array([X1,
                       X2,
                       X3,
                       X4,
                       X5]).reshape(1,-1)
                       
    # Prediction is calculated based on the input_X array
    prediction = model.predict(input_X)[0]
    
    # And retuned to the Output of the callback function
    return "Prediction: {}".format(int(prediction))

if __name__ == "__main__":
    app.run_server(debug=True)