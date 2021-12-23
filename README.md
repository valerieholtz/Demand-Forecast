# Demand-Forecast

Machine Learning based dashboard using a Linear Regression model for predicting demand for bike rentals in the Capital Bikeshare program in Washington, D.C.

<img src="https://user-images.githubusercontent.com/79086000/147233912-b99a40be-476f-4965-a8f7-8b1c77f6ce36.gif">


![Dashboard](https://user-images.githubusercontent.com/79086000/147233912-b99a40be-476f-4965-a8f7-8b1c77f6ce36.gif)

An approach to Machine Learning with Linear Regression. Through Explorative Data Analysis and Feature Engineering I was able to construct robust models to estimate number of bike rentals. OLS and Recursive Feature Elimination was used to pick the best features for the Linear model. GridSearchCV was used to attain the best fitting model and parameters.

### Guideline questions for the EDA
- Which weather conditions and time criteria impact registered users to use bike sharing company's rental bicycles as registered users?

### Summary of the OLS Regression results:

<img src="https://user-images.githubusercontent.com/79086000/147234098-c9c9ae68-e0b0-4d4f-aa4d-0cbc272249b1.png" width="600">

### Models used:
- Linear Regression
- Polynomial Regression
- Ridge
- Lasso

The model with the overall pest performance was Ridge with the following scores:

![score](https://user-images.githubusercontent.com/79086000/147234211-34e52884-2a32-44b8-ade5-53266b9220b2.png)

### Data source
https://www.kaggle.com/lakshmi25npathi/bike-sharing-dataset

### Tech used:
- Python
- Scikit-learn
- GridSearchCV
- Dash
- Plotly
- HTML
- CSS
