import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# verilerin alınması ve işlenmesi
data = pd.read_csv('teknoloji.csv')
#print(data)
#print(data.isnull().sum())


Y = data.iloc[:,[4]]

Ulkeadi  =data.iloc[:,0]
yil = data.iloc[:,1]
teknolojitutari = data.iloc[:,2]
uretimmiktarı = data.iloc[:,3]
maliyetfiyati = data.iloc[:,-1]

#kategorik verilerin encode işleminden geçirilmesi
le = LabelEncoder()
Ulkeadi= le.fit_transform(Ulkeadi)
#print(Ulkeadi)

ohe = OneHotEncoder()
Ulkeadi = ohe.fit_transform(Ulkeadi.reshape(-1,1)).toarray()
#print(Ulkeadi)


le = LabelEncoder()
teknolojitutari= le.fit_transform(teknolojitutari)
#print(teknolojitutari)

ohe = OneHotEncoder()
teknolojitutari = ohe.fit_transform(teknolojitutari.reshape(-1,1)).toarray()
#print(teknolojitutari)

# encode edilmiş verilerden dataframe oluşturulması
df1 = pd.DataFrame(data=Ulkeadi, index= range(126), columns=['Almanya','Turkiye','Italya','Fransa'])

df2 = pd.DataFrame(data=teknolojitutari, index=range(126), columns=['Akilli Cihazlar','Otonom Araçlar','Drone','Yuz tanima teknolojisi'])

s1 = pd.concat([df1, df2], axis=1)
s2 = pd.concat([s1, yil], axis=1)
s3 = pd.concat([maliyetfiyati, uretimmiktarı], axis=1)
s4 = pd.concat([s2, s3], axis=1)
s5 = pd.concat([s4, Y], axis=1)

# Verilerin normalize edilmesi
min_max_scale = MinMaxScaler()
scaled_data = min_max_scale.fit_transform(s5)

# Bağımsız değişkenler ve hedef değişken ayrıştırması
X = scaled_data[:,:-1]
y = scaled_data[:,-1]

# K-fold çapraz doğrulama için ayarları belirleme ve modelin performansını değerlendirme
r2_scores = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Verilerin standardize edilmesi
    std_scale = StandardScaler()
    X_train = std_scale.fit_transform(X_train)
    X_test = std_scale.transform(X_test)

    # Modelin eğitimi ve tahmini
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # R2 skoru hesaplama ve kaydetme
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# R2 skorlarının ortalamasını alarak genel performansı hesaplama
mean_r2 = np.mean(r2_scores)

print("Ortalama R2 skoru:", mean_r2)