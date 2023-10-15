"""
Ref : https://www.kaggle.com/code/mathieucaron/early-readmission-prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def map_diagnosis(data, cols):
    for col in cols:
        data.loc[(data[col].str.contains("V")) | (data[col].str.contains("E")), col] = -1
        data[col] = data[col].astype(np.float16)

    for col in cols:
        data["temp_diag"] = np.nan
        data.loc[(data[col]>=390) & (data[col]<=459) | (data[col]==785), "temp_diag"] = "Circulatory"
        data.loc[(data[col]>=460) & (data[col]<=519) | (data[col]==786), "temp_diag"] = "Respiratory"
        data.loc[(data[col]>=520) & (data[col]<=579) | (data[col]==787), "temp_diag"] = "Digestive"
        data.loc[(data[col]>=250) & (data[col]<251), "temp_diag"] = "Diabetes"
        data.loc[(data[col]>=800) & (data[col]<=999), "temp_diag"] = "Injury"
        data.loc[(data[col]>=710) & (data[col]<=739), "temp_diag"] = "Muscoloskeletal"
        data.loc[(data[col]>=580) & (data[col]<=629) | (data[col] == 788), "temp_diag"] = "Genitourinary"
        data.loc[(data[col]>=140) & (data[col]<=239), "temp_diag"] = "Neoplasms"

        data["temp_diag"] = data["temp_diag"].fillna("Other")
        data[col] = data["temp_diag"]
        data = data.drop("temp_diag", axis=1)
    return data

def preprocess_data(data):
    data = data.replace("?",np.nan)
    data = data.replace({"NO":0,
                         "<30":1,
                         ">30":0})

    mapped_race = {"Asian":"Other","Hispanic":"Other"}
    data.race = data.race.replace(mapped_race)

    # Dropping Unknown gender since it is only just 1 sample
    data = data.drop(data.loc[data["gender"]=="Unknown/Invalid"].index, axis=0)

    data.age = data.age.replace({"[70-80)":75,
                             "[60-70)":65,
                             "[50-60)":55,
                             "[80-90)":85,
                             "[40-50)":45,
                             "[30-40)":35,
                             "[90-100)":95,
                             "[20-30)":25,
                             "[10-20)":15,
                             "[0-10)":5})

    mapped = {1.0:"Emergency",
              2.0:"Emergency",
              3.0:"Elective",
              4.0:"New Born",
              5.0:np.nan,
              6.0:np.nan,
              7.0:"Trauma Center",
              8.0:np.nan}

    data.admission_type_id = data.admission_type_id.replace(mapped)

    mapped_discharge = {1:"Discharged to Home",
                        6:"Discharged to Home",
                        8:"Discharged to Home",
                        13:"Discharged to Home",
                        19:"Discharged to Home",
                        18:np.nan,25:np.nan,26:np.nan,
                        2:"Other",3:"Other",4:"Other",
                        5:"Other",7:"Other",9:"Other",
                        10:"Other",11:"Other",12:"Other",
                        14:"Other",15:"Other",16:"Other",
                        17:"Other",20:"Other",21:"Other",
                        22:"Other",23:"Other",24:"Other",
                        27:"Other",28:"Other",29:"Other",30:"Other"}

    data["discharge_disposition_id"] = data["discharge_disposition_id"].replace(mapped_discharge)

    mapped_adm = {1:"Referral",2:"Referral",3:"Referral",
                  4:"Other",5:"Other",6:"Other",10:"Other",22:"Other",25:"Other",
                  9:"Other",8:"Other",14:"Other",13:"Other",11:"Other",
                  15:np.nan,17:np.nan,20:np.nan,21:np.nan,
                  7:"Emergency"}
    data.admission_source_id = data.admission_source_id.replace(mapped_adm)

    data = map_diagnosis(data,["diag_1","diag_2","diag_3"])

    data.change = data.change.replace("Ch","Yes")

    data["max_glu_serum"] = data["max_glu_serum"].replace({">200":2,
                                                            ">300":2,
                                                            "Norm":1,
                                                            "None":0}) 

    data["A1Cresult"] = data["A1Cresult"].replace({">7":2,
                                               ">8":2,
                                               "Norm":1,
                                               "None":0})

    data['race'] = data['race'].fillna(data['race'].mode()[0])
    data['admission_type_id'] = data['admission_type_id'].fillna(data['admission_type_id'].mode()[0])
    data['discharge_disposition_id'] = data['discharge_disposition_id'].fillna(data['discharge_disposition_id'].mode()[0])
    data['admission_source_id'] = data['admission_source_id'].fillna(data['admission_source_id'].mode()[0])
    cat_data = data.select_dtypes('O')
    num_data = data.select_dtypes(np.number)
    le = LabelEncoder()
    for i in cat_data:
        cat_data[i] = le.fit_transform(cat_data[i])
    data = pd.concat([num_data,cat_data],axis=1)
    print("There are {} unique patients in the data".format(data.patient_nbr.unique().shape[0]))
    data_percentages = data["readmitted"].value_counts(normalize=True) * 100
    plt.figure(figsize=(3,2))
    sns.barplot(x=data_percentages.index, y=data_percentages.values)
    plt.ylabel('Percentage')
    plt.xlabel('Readmitted')
    plt.title('% of Hospital readmission')
    plt.show()
    return data
