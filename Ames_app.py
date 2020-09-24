import numpy as np
import pandas as pd
from rdkit import Chem
from PIL import Image
import joblib
import pickle
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
import streamlit as st


######################
# Custom function
######################
## Calculate molecular descriptors

def Calculate (smiles, verbose = False):
    mols = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        mols.append(mol)
    
    baseData= np.arange(1,1)
    i=0
    for mol in mols:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_TPSA = Descriptors.TPSA(mol)

        row = np.array([desc_MolLogP,
                       desc_TPSA])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP", "TPSA"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

######################
# Page Title
######################

image = Image.open('AMES_logo.jpg')

st.image(image, use_column_width=True)

st.write("""# Adnane's Website : Ames mutagenicity Prediction Website This website predicts the *AMES test* behavior of molecules !""")


######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input Features')

## Read SMILES input
SMILES_input = "CCCCO\nc1ccccc1\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:] # Skips the dummy first item

## Calculate molecular descriptors
#st.header('Computed molecular descriptors')
X = Calculate(SMILES)
#X[1:] # Skips the dummy first item

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('Ames_calssifcator.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(X)
#prediction_proba = load_model.predict_proba(X)

st.header('Predicted Mutagenicity')
prediction[1:] # Skips the dummy first item 
