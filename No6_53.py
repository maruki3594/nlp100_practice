import pandas as pd
import pickle

def Encoder(sign):
    if sign == "e":
        code = 0
    elif sign == "b":
        code = 1
    elif sign == "m":
        code = 2
    elif sign == "t":
        code = 3
    else:
        pass
    return code

def Decoder(code):
    if code == 0:
        sign = "e"
    elif code == 1:
        sign = "b"
    elif code == 2:
        sign = "m"
    elif code == 3:
        sign = "t"
    else:
        pass
    return sign

df = pd.read_csv("./test.feature.txt", sep='\t', index_col = 0)
df_y = df["CATEGORY"].map(Encoder)
X = df.iloc[:,2:].values.tolist()
Y = df_y.values.tolist()
lr = pickle.load(open("./Logistic_model.sav", 'rb'))
proba = lr.predict_proba(X)

df_proba = pd.DataFrame()
df_proba["TITLE"] = df["TITLE"]
df_proba_set = pd.DataFrame(proba)
print(df_proba_set)
df_proba["CATEGORY"] = df_proba_set.idxmax(axis=1).map(Decoder)
df_proba["PROBA"] = df_proba_set.max(axis=1)

# print(df_proba)