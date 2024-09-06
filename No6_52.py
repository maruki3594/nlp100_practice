from sklearn.linear_model import LogisticRegression
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

df_tr = pd.read_csv("./train.feature.txt", index_col = 0, sep='\t')
df_tr["CATEGORY"] = df_tr["CATEGORY"].map(Encoder)
print(df_tr['CATEGORY'])
# print(df_tr['CATEGORY'])
X = df_tr.iloc[:,2:].values.tolist()
Y = df_tr["CATEGORY"].values.tolist()
lr = LogisticRegression()
lr.fit(X, Y)
pickle.dump(lr, open("./Logistic_model.sav", 'wb'))
