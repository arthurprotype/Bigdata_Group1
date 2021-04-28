import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split

label_dic = {
    'normal': (1, 0, 0),
    'inter': (0, 1, 0),
    'ictal': (0, 0, 1),
}

# 读取目标文件夹中的所有文件
def folder_to_df(letter):
    full_path ="EEG_data/"+ letter + "/*.*"
    files = glob.glob (full_path)
    df_list = []
    for file in files:
        df_list.append (pd.read_csv (file, header = None))
    big_df = pd.concat (df_list, ignore_index=True, axis= 1)

    return big_df.T

# 窗口切割
def window (a, w = 256, o = 128):
    view = []
    for i in range (0, a.shape[0] - w, o):
        sub = a[i:i+w]
        """
        MinMax = np.ptp (sub)
        sub = (sub - np.mean (sub)) / np.std (sub)
        sub = list (sub)
        sub.append (MinMax)
        """
        view += [sub]
    
    return view.copy()

# 数据扩充
def enrich_data (df): 
    res = []
    for i in range (df.shape[0]):
        res += [window (df.iloc[i].values)]
    return res

# 读取所有的文件夹
def load_data_as_df():
    A = folder_to_df('Z')
    B = folder_to_df('O')
    C = folder_to_df('N')
    D = folder_to_df('F')
    E = folder_to_df('S')
    
    normal = A.append (B).reset_index(drop = True)
    interictal = C.append (D).reset_index(drop = True)
    ictal = E

    return normal, interictal, ictal

# 数据扩充
def format_enrich_data(normal, interictal, ictal):
    normal_data_enr = np.asarray (enrich_data(normal)).reshape(-1, np.asarray(enrich_data(normal)).shape[-1])
    interictal_data_enr = np.asarray (enrich_data(interictal)).reshape(-1, np.asarray(enrich_data(interictal)).shape[-1])
    ictal_data_enr = np.asarray (enrich_data(ictal)).reshape(-1, np.asarray(enrich_data(ictal)).shape[-1])

    # 更换数据形式
    normal_data_enr_df = pd.DataFrame(normal_data_enr)
    interictal_data_enr_df = pd.DataFrame(interictal_data_enr)
    ictal_data_enr_df = pd.DataFrame(ictal_data_enr)
    
    # 打标签
    normal_data_enr_lab = pd.DataFrame ([label_dic['normal'] for _ in range (normal_data_enr.shape[0])])
    interictal_data_enr_lab = pd.DataFrame ([label_dic['inter'] for _ in range (interictal_data_enr.shape[0])])
    ictal_data_enr_lab = pd.DataFrame ([label_dic['ictal'] for _ in range (ictal_data_enr.shape[0])])

    # 拼接并划分出数据与标签
    data = pd.concat([normal_data_enr_df, interictal_data_enr_df, ictal_data_enr_df], ignore_index = True)
    labels = pd.concat([normal_data_enr_lab, interictal_data_enr_lab, ictal_data_enr_lab], ignore_index = True)
    
    return data.values, labels.values

def rnn_transform (raw_dat, frame_size = 1):
    if raw_dat.shape[1] % frame_size == 0:
        return raw_dat.reshape (raw_dat.shape[0], -1, frame_size)
    else:
        print ("error frame_size")
        exit (1)

normal, interictal, ictal = load_data_as_df()

normal_train, normal_test = train_test_split(normal, test_size = 0.2)
interictal_train, interictal_test = train_test_split(interictal, test_size = 0.2)
ictal_train, ictal_test = train_test_split(ictal, test_size = 0.2)

X_train, y_train = format_enrich_data (normal_train, interictal_train, ictal_train)
X_test, y_test = format_enrich_data (normal_test, interictal_test, ictal_test)

print (y_test)
"""#  !!!!!!!!!!!!!!! rnn/lstm/gru 这两行取消注释 !!!!!!!!!!!!!! cnn/ann 删掉即可 !!!!!!!!!!!!!!!!!!!!!
X_train = rnn_transform (X_train)
X_test = rnn_transform (X_test)
"""