import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# ====== 1. Load Data ======
pathspectra = "D:\\PYTHON\\PYTHON SAPI BABII\\DATA XLS\\Data Spektra SB.xlsx"
pathwave = "D:\\PYTHON\\PYTHON SAPI BABII\\DATA XLS\\Wavelength SB.xlsx"

xdata = pd.read_excel(pathspectra)
wdata = pd.read_excel(pathwave)

spect = xdata.values[:, 1:]  # Mengabaikan kolom pertama (misal non-wavelength)
wavelength = wdata.values[:, 0]  # Mengambil panjang gelombang

# ====== 2. PCA Analysis ======
PCn = 10  # Menentukan jumlah komponen utama maksimum
mdl = PCA(n_components=PCn)
XS = mdl.fit_transform(spect)  # Hasil PCA
XL = mdl.components_  # PC Loadings
per_var = mdl.explained_variance_ratio_ * 100  # Explained Variance (%)

# Tampilkan explained variance tiap PC
print("Explained Variance Ratio (%):")
for i, var in enumerate(per_var, start=1):
    print(f'PC{i}: {var:.2f}%')

# ====== 3. Scree Plot ======
plt.figure(figsize=(6,4))
plt.plot(range(1, PCn+1), per_var, marker='o', linestyle='-', color='b')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot - PCA')
plt.grid()
plt.show()

# ====== 4. Cumulative Explained Variance ======
cumulative_var = np.cumsum(per_var)
num_pc_95 = np.argmax(cumulative_var >= 95) + 1  # PC untuk menjelaskan ≥95% varians

print(f"Jumlah PC untuk menjelaskan ≥95% varians: {num_pc_95}")

# Plot Cumulative Variance
plt.figure(figsize=(6,4))
plt.plot(range(1, PCn+1), cumulative_var, marker='o', linestyle='-', color='r')
plt.axhline(y=95, color='gray', linestyle='--', label="95% Threshold")
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Variance - PCA')
plt.legend()
plt.grid()
plt.show()

# ====== 5. PCA Scatter Plot (2D & 3D) ======
group_sizes = 10  # Ubah sesuai jumlah sampel per grup
percentages = ["1%", "3%", "5%", "7%", "10%", "15%", "20%", "30%", "40%", "50%", 
               "60%", "70%", "80%", "85%", "90%", "93%", "95%", "97%", "99%"]
colors = plt.cm.viridis(np.linspace(0, 1, len(percentages)))

# 2D PCA Plot
fig1 = plt.figure(figsize=(6,6))
for i, (color, label) in enumerate(zip(colors, percentages)):
    start_idx = i * group_sizes
    end_idx = (i + 1) * group_sizes
    plt.scatter(XS[start_idx:end_idx, 0], XS[start_idx:end_idx, 1], color=color, edgecolors='k', label=label)

plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel(f'PC1: {per_var[0]:.2f}%')
plt.ylabel(f'PC2: {per_var[1]:.2f}%')
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Adulterasi (%)")
plt.title('2D Scatter Plot PCA')
plt.show()

# 3D PCA Plot
fig2 = plt.figure(figsize=(6,6))
ax = fig2.add_subplot(111, projection='3d')

for i, (color, label) in enumerate(zip(colors, percentages)):
    start_idx = i * group_sizes
    end_idx = (i + 1) * group_sizes
    ax.scatter(XS[start_idx:end_idx, 0], XS[start_idx:end_idx, 1], XS[start_idx:end_idx, 2], 
               color=color, edgecolors='k', label=label)

ax.set_xlabel(f'PC1: {per_var[0]:.2f}%')
ax.set_ylabel(f'PC2: {per_var[1]:.2f}%')
ax.set_zlabel(f'PC3: {per_var[2]:.2f}%')
ax.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Adulterasi (%)")
plt.title('3D Scatter Plot PCA')
plt.show()

# ====== 6. PC Loading Plot ======
PC1 = XL[0, :]
PC2 = XL[1, :]
PC3 = XL[2, :]

plt.figure(figsize=(6,4))
plt.plot(wavelength, PC1, color='r', label='PC1')
plt.plot(wavelength, PC2, color='g', label='PC2')
plt.plot(wavelength, PC3, color='b', label='PC3')
plt.xlabel('Wavelength (nm)')
plt.ylabel('PC Loading Value')
plt.legend(loc='best')
plt.title('PC Loading Plot')
plt.grid()
plt.show()