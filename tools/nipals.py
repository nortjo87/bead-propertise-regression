import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sys import stdout

def pls_NIPALS(x_cal, y_cal, x_val, y_val,lv=10,ns=5):
    sr, sc = np.shape(x_cal)
    max_LVs = min(lv, sc)  # Batasi maksimal 10 komponen
    comp = np.arange(1, max_LVs + 1)
    mse = []
    
    # Gunakan KFold CV (lebih stabil daripada LOO)
    cv = KFold(n_splits=ns, shuffle=True, random_state=42)
    
    # Cari LV optimal
    for i in comp:
        pls = PLSRegression(n_components=i, scale=True)  # Gunakan scaling
        y_cv = cross_val_predict(pls, x_cal, y_cal, cv=cv)
        mse.append(mean_squared_error(y_cal, y_cv))
        stdout.write(f"\r{int((i/max_LVs)*100)}% Processing LVs...")
        stdout.flush()
    stdout.write("\n")
    
    # Tentukan jumlah LV terbaik
    loc = np.argmin(mse)
    LVs = loc + 1

    # Model final
    pls2 = PLSRegression(n_components=LVs, scale=True)
    pls2.fit(x_cal, y_cal)
    Beta = pls2.coef_
    eps = pls2._y_mean - np.dot(pls2._x_mean, Beta.T)
    
    # Prediksi
    C_Y_Value = eps + np.dot(x_cal, Beta.T)
    V_Y_Value = eps + np.dot(x_val, Beta.T)
    
    # Evaluasi
    R2C = r2_score(y_cal, C_Y_Value)
    R2P = r2_score(y_val, V_Y_Value)
    V_Y_Prediction1 = cross_val_predict(pls2, x_cal, y_cal, cv=cv)
    R2CV = r2_score(y_cal, V_Y_Prediction1)
    
    SEC = np.sqrt(mean_squared_error(y_cal, C_Y_Value))
    SECV = np.sqrt(mean_squared_error(y_cal, V_Y_Prediction1))
    SEP = np.sqrt(mean_squared_error(y_val, V_Y_Value))
    RPD = np.std(y_val) / SEP if SEP != 0 else None
    
    # Plot RMSECV agar mudah cek overfitting
    plt.figure(figsize=(6,4))
    plt.plot(comp, np.sqrt(mse), marker='o')
    plt.xlabel('Number of Latent Variables (LVs)')
    plt.ylabel('RMSECV')
    plt.title('PLSR Model Selection')
    plt.grid(True)
    plt.show()
    
    return LVs, Beta, eps, R2C, SEC, R2CV, SECV, R2P, SEP, RPD, C_Y_Value, V_Y_Prediction1, V_Y_Value, pls2
