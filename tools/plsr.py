import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NIPALS import pls_NIPALS
from PREPROCESSING import range_norm, mean_norm, max_norm, msc, snv, sg1, sg2, smoothing_mean

# Customize font
plt.rc('font', family='Serif', size=13)

path = "C:\\Data Bunda\\BUNDA\\PYTHON NADIA DUA MEI\\BIJI BASAH NIR ALL DATA\\KADAR AIR\\RESULT\\"
sample_name = "PLSBIJIBASAHKADARAIRNIR"

# File paths = lokasi spesifik di mana file atau folder disimpan dalam sistem komputer
cal_path = path+"KA_CAL70.xlsx"
val_path = path+"KA_VAL30.xlsx"
wav_path = path+"Wavelenght NIR.xlsx"

# Read Excel files
cal = pd.read_excel(cal_path)
val = pd.read_excel(val_path)
wav = pd.read_excel(wav_path)

# Extract data
x_cal = cal.values[1:, 1:]
y_cal = cal.values[1:, 0]
x_val = val.values[1:, 1:]
y_val = val.values[1:, 0]
wavebands = wav.values[0:, 0]
print(wavebands.shape)

# Preprocessing methods
preprocessing_methods = [1,2,3,4,5,6,7,8,9]
results = []

# Apply preprocessing methods
for PR in preprocessing_methods:
    if PR == 1:
        input_cal = mean_norm(x_cal)
        input_val = mean_norm(x_val)
    elif PR == 2:
        input_cal = max_norm(x_cal)
        input_val = max_norm(x_val)
    elif PR == 3:
        input_cal = range_norm(x_cal)
        input_val = range_norm(x_val)
    elif PR == 4:
        input_cal = msc(x_cal)
        input_val = msc(x_val)
    elif PR == 5:
        input_cal = snv(x_cal)
        input_val = snv(x_val)
    elif PR == 6:
        input_cal = sg1(x_cal)
        input_val = sg1(x_val)
    elif PR == 7:
        input_cal = sg2(x_cal)
        input_val = sg2(x_val)
    elif PR == 8:
        input_cal = x_cal
        input_val = x_val
    elif PR == 9:
        mov_wind = 8  # Adjust window size as needed
        input_cal = smoothing_mean(x_cal, mov_wind)
        input_val = smoothing_mean(x_val, mov_wind)
    else:
        raise ValueError("Invalid preprocessing number. Choose a number between 1 and 9")

    input_y_cal = y_cal
    input_y_val = y_val

    # Apply PLS NIPALS
    LVs, Beta3, eps3, R2C, SEC, R2CV, SECV, R2P, SEP, RPD, C_Y_Value, V_Y_Prediction1, V_Y_Value = pls_NIPALS(input_cal, input_y_cal, input_val, input_y_val)

    # Collect results
    result = [PR, R2C, SEC, R2CV, SECV, R2P, SEP, RPD, LVs]
    results.append(result)

# Save results to Excel
cols = ['Prepro-', 'R2C', 'SEC', 'R2CV', 'SECV', 'R2P', 'SEP', 'RPD', 'LVs']
df = pd.DataFrame(results, columns=cols)
result_name = f"{path}{sample_name}.xlsx"
df.to_excel(result_name, index=False)

# Save predicted values
cc_name =  f"{path}cc_{sample_name}.xlsx"
pp_name = f"{path}pp_{sample_name}.xlsx"
pd.DataFrame(C_Y_Value).to_excel(cc_name, index=False)
pd.DataFrame(V_Y_Value).to_excel(pp_name, index=False)

# Plot calibration and validation results
fig1 = plt.figure(figsize=(5, 5))
plt.scatter(y_cal, C_Y_Value, edgecolors='k', color='b', linewidths=1, label='Calibration')
plt.scatter(y_val, V_Y_Value, edgecolors='k', color='g', linewidths=1, label='Validation')
plt.plot(y_val, y_val, color='k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
sctr_name = f"{path}{sample_name}_sctr.png"
fig1.savefig(sctr_name, dpi=650)
plt.title('Calibration vs Validation')

# Smoothing Beta coefficients and save
Beta = smoothing_mean(Beta3, 50)
Beta_name = f"{path}{sample_name}_Beta.xlsx"
pd.DataFrame(Beta.T).to_excel(Beta_name, index=False)

# Plot Beta coefficients
fig2 = plt.figure()
plt.plot(wavebands, Beta.T, color='m', linewidth=2)
plt.xlim([950, 1700])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Beta coefficient value')
Beta_ir_name = f"{path}{sample_name}Beta_ir.png"
fig2.savefig(Beta_ir_name, dpi=500)
plt.title('Beta Coefficients')

# Menampilkan plot
plt.show()