from sklearn.metrics import accuracy_score
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

lr = pickle.load(open("./Logistic_model.sav", 'rb'))

# df = pd.read_csv("./train.feature.txt", sep='\t', index_col = 0)
# df_y = df["CATEGORY"].map(Encoder)
# X_train = df.iloc[:,2:].values.tolist()
# Y_train = df_y.values.tolist()
# Y_pred_train = lr.predict(X_train)
# print("訓練データ", accuracy_score(y_true = Y_train, y_pred = Y_pred_train))
# pickle.dump(Y_pred_train, open('./Y_pred_train.sav', "wb"))
# with open('Y_pred_train', mode='w') as f:
#     f.write((Y_pred_train))

df = pd.read_csv("./test.feature.txt", sep='\t', index_col = 0)
df_y = df["CATEGORY"].map(Encoder)
X_test = df.iloc[:,2:].values.tolist()
Y_test = df_y.values.tolist()
Y_pred_test = lr.predict(X_test)
print("評価データ", accuracy_score(y_true = Y_test, y_pred = Y_pred_test))
pickle.dump(Y_pred_test, open('./Y_pred_test.sav', 'wb'))
