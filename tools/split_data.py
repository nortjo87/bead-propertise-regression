import numpy as np
import pandas as pd

def datasplit_simple(df, cs):
    # df = pd.read_excel(pathspectra)
    data = df.values
    np.random.seed(42)
    shuffled_idx = np.random.permutation(data.shape[0])
    split_idx = int(cs * data.shape[0])

    cal_data = data[shuffled_idx[:split_idx], :]
    val_data = data[shuffled_idx[split_idx:], :]

    print("Data has been separated!")
    return cal_data, val_data

# -------------------- RUN THE FUNCTION -----------------------------#
# pathspectra = "C:\\Data Bunda\\BUNDA\\PYTHON NADIA DUA MEI\\BIJI BASAH NIR ALL DATA\\KADAR AIR\\DATA XLS\\DATA RAW_NIR_BASAH_KADAR AIR.xlsx"
# pathwave = "C:\\Data Bunda\\BUNDA\\PYTHON NADIA DUA MEI\\BIJI BASAH NIR ALL DATA\\KADAR AIR\\DATA XLS\\Wavelenght NIR.xlsx"

# wave = pd.read_excel(pathwave)
# wave1 = wave.iloc[:, 0].tolist()
# col = ['VALUE'] + wave1
# CS = 0.70

# Cal, Val = datasplit_simple(pathspectra, CS)

# Cal2 = pd.DataFrame(Cal, columns=col)
# Val2 = pd.DataFrame(Val, columns=col)
# Cal2.to_excel("C:\\Data Bunda\\BUNDA\\PYTHON NADIA DUA MEI\\BIJI BASAH NIR ALL DATA\\KADAR AIR\\RESULT\\KA_CAL70.xlsx", index=False)
# Val2.to_excel("C:\\Data Bunda\\BUNDA\\PYTHON NADIA DUA MEI\\BIJI BASAH NIR ALL DATA\\KADAR AIR\\RESULT\\KA_VAL30.xlsx",index=False)