import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import xgboost as xgb
import shap
import pickle

image = Image.open('Flowchart.jpg')

st.image(image, use_column_width=True)

st.write("""
# Expalinable XGBoost Prediction of Axial Load-Carrying Capacity of FRP-RC Columns

This app predict the axial load-carrying capacity of FRP-RC columns

***
""")

# Loads the FRP-RC columns Dataset
frprccolumns = pd.read_excel("FRP-RC_Columns_Database.xlsx", usecols="A:AA", header=0)

# Convert data
frprccolumns['LamdaC'] = frprccolumns['LamdaC'].astype(float)
frprccolumns['SpacPitch'] = frprccolumns['SpacPitch'].astype(float)
frprccolumns['EoverD'] = frprccolumns['EoverD'].astype(float)
frprccolumns['ffuL'] = frprccolumns['ffuL'].astype(float)
frprccolumns['Ag'] = frprccolumns['Ag'].astype(float)
frprccolumns['fcp'] = frprccolumns['fcp'].astype(float)
frprccolumns['RhoEf'] = frprccolumns['RhoEf'].astype(float)
frprccolumns['Pexp'] = frprccolumns['Pexp'].astype(float)
frprccolumns['EfrpL'] = frprccolumns['EfrpL'].astype(float)

frprccolumns = frprccolumns[['LamdaC', 'Circular', 'Ag', 'TypeCon', 'fcp', 'TypeL', 'RhoEf',
                             'EfrpL', 'ffuL', 'TypeH', 'Config', 'SpacPitch', 'EoverD', 'Pexp']]
y = frprccolumns['Pexp'].copy()
X = frprccolumns.drop('Pexp', axis=1).copy()

X_encoded = pd.get_dummies(X, columns=['Circular',
                                       'TypeCon',
                                       'TypeL',
                                       'TypeH',
                                       'Config'], drop_first=True)

# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
value = ("0", "1")
options = list(range(len(value)))


def input_variable():
    LamdaC = st.sidebar.slider('Slenderness ratio', float(X_encoded.LamdaC.min()), float(X_encoded.LamdaC.max()),
                               float(X_encoded.LamdaC.mean()))
    Ag = st.sidebar.slider('Column gross sectional area, (mm2)', float(X_encoded.Ag.min()), float(X_encoded.Ag.max()),
                           float(X_encoded.Ag.mean()))
    fcp = st.sidebar.slider('Compressive strength of concrete, (MPa)', float(X_encoded.fcp.min()), float(X_encoded.fcp.max()),
                            float(X_encoded.fcp.mean()))
    RhoEf = st.sidebar.slider('Longitudinal reinforcement ratio, RhoEf (%)', float(X_encoded.RhoEf.min()),
                              float(X_encoded.RhoEf.max()), float(X_encoded.RhoEf.mean()))
    EfrpL = st.sidebar.slider('Modulus of elasticity of FRP reinforcement, (GPa)', float(X_encoded.EfrpL.min()),
                              float(X_encoded.EfrpL.max()), float(X_encoded.EfrpL.mean()))
    ffuL = st.sidebar.slider('Ultimate strength of FRP reinforcement, (MPa) ', float(X_encoded.ffuL.min()),
                             float(X_encoded.ffuL.max()), float(X_encoded.ffuL.mean()))
    SpacPitch = st.sidebar.slider('Spacing/pitch of transversal reinforcement, (mm) ', float(X_encoded.SpacPitch.min()),
                                  float(X_encoded.SpacPitch.max()), float(X_encoded.SpacPitch.mean()))
    EoverD = st.sidebar.slider('Eccentricity ratio (%)', float(X_encoded.EoverD.min()), float(X_encoded.EoverD.max()),
                               float(X_encoded.EoverD.mean()))
    Circular = st.sidebar.radio('Column section type', ('Rectangular/Square', 'Circular'))
    TypeCon = st.sidebar.radio('Concrete type', ('NWC', 'LWC', 'GC'))
    TypeL = st.sidebar.radio('Type of longitudinal reinforcement', ('GFRP', 'CFRP', 'BFRP'))
    TypeH = st.sidebar.radio('Type of transversal reinforcement ', ('GFRP', 'CFRP', 'BFRP', 'Steel'))
    Config = st.sidebar.radio('Configuration of transversal reinforcement', ('Spiral', 'Ties', 'Hoops'))

    if Circular == 'Rectangular/Square':
        Circular_Yes = 0
    else:
        Circular_Yes = 1

    if TypeCon == 'NWC':
        TypeCon_NWC = 1
        TypeCon_LWC = 0
    elif TypeCon == 'LWC':
        TypeCon_LWC = 1
        TypeCon_NWC = 0
    else:
        TypeCon_LWC = 0
        TypeCon_NWC = 0

    if TypeL == 'GFRP':
        TypeL_GFRP = 1
        TypeL_CFRP = 0
    elif TypeL == 'CFRP':
        TypeL_GFRP = 0
        TypeL_CFRP = 1
    else:
        TypeL_GFRP = 0
        TypeL_CFRP = 0

    if TypeH == 'GFRP':
        TypeH_GFRP = 1
        TypeH_CFRP = 0
        TypeH_Steel = 0
    elif TypeH == 'CFRP':
        TypeH_GFRP = 0
        TypeH_CFRP = 1
        TypeH_Steel = 0
    elif TypeH == 'Steel':
        TypeH_GFRP = 0
        TypeH_CFRP = 0
        TypeH_Steel = 1
    else:
        TypeH_GFRP = 0
        TypeH_CFRP = 0
        TypeH_Steel = 0

    if Config == 'Spiral':
        Config_Spiral = 1
        Config_Ties = 0
    elif Config == 'Ties':
        Config_Spiral = 0
        Config_Ties = 1
    else:
        Config_Spiral = 0
        Config_Ties = 0

    data = {'LamdaC': LamdaC,
            'Ag': Ag,
            'fcp': fcp,
            'RhoEf': RhoEf,
            'EfrpL': EfrpL,
            'ffuL': ffuL,
            'SpacPitch': SpacPitch,
            'EoverD': EoverD,
            'Circular_Yes': Circular_Yes,
            'TypeCon_LWC': TypeCon_LWC,
            'TypeCon_NWC': TypeCon_NWC,
            'TypeL_CFRP': TypeL_CFRP,
            'TypeL_GFRP': TypeL_GFRP,
            'TypeH_CFRP': TypeH_CFRP,
            'TypeH_GFRP': TypeH_GFRP,
            'TypeH_Steel': TypeH_Steel,
            'Config_Spiral': Config_Spiral,
            'Config_Ties': Config_Ties
            }

    features = pd.DataFrame(data, index=[0])
    return features

df = input_variable()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

xgb_model = pickle.load(open('FRPRC_xgb_model.pkl', 'rb'))

prediction = xgb_model.predict(df)

st.header('Predicted maximun axial load')
st.write('Pmax =', prediction[0])
st.write('---')

# Explaining the model's predictions using SHAP values

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_encoded)

st.header('Interpretation of XGBoost prediction model using SHAP values')

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title('SHAP summary plot', fontweight="bold", fontsize=16, pad=20)
shap.summary_plot(shap_values, X_encoded)
st.pyplot(fig, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_title('Relative importance for each feature', fontweight="bold", fontsize=16, pad=20)
shap.summary_plot(shap_values, X_encoded, plot_type="bar")
st.pyplot(fig, bbox_inches='tight')
