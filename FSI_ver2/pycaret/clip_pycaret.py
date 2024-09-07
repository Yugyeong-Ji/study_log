# Import Library
# 제출 파일 생성 관련
import os
import zipfile

# 데이터 처리 및 분석
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# 머신러닝 전처리
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# 머신러닝 모델
import xgboost as xgb

# 합성 데이터 생성
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# To ignore all warnings
import warnings
warnings.filterwarnings('ignore')

# pycaret AutoML 사용
import pycaret
from pycaret.classification import *
from pycaret.classification import ClassificationExperiment

print(pycaret.__version__)
# 생성 🏭
# Load Data
train_all = pd.read_csv("/workspace/Dataset/FSI/clip_downloads/FSI/train_features.csv")
test_all = pd.read_csv("/workspace/Dataset/FSI/clip_downloads/FSI/test_features.csv")
all_synthetic_data = pd.read_csv("/workspace/Dataset/FSI/clip_downloads/FSI/generated_data_features.csv")



## 원본 데이터와 concat
origin_train = train_all.drop(columns="ID")
all_synthetic_data = all_synthetic_data.drop(columns="ID")
train_total = pd.concat([origin_train, all_synthetic_data], ignore_index=True)
train_total.shape
train_total.head()



# 폴더 생성 및 작업 디렉토리 변경
os.makedirs('/workspace/Dataset/FSI/clip_downloads/submission', exist_ok=True)
os.chdir("/workspace/Dataset/FSI/clip_downloads/submission")



# Data Preprocessing : pycaret setup / pycaret exp(init class)
exp = ClassificationExperiment()
print(type(exp))

exp.setup(train_total, target = 'Fraud_Type', session_id = 123) #setup data for pycaret experiment


# Compare Models : 모델 선정(autoML)
print(exp.models())
best = exp.compare_models(include = ['xgboost']) #compare models by OOP method(nothing different from compare_models() function)


# save pipeline
exp.save_model(best, '/workspace/Dacon_FSI/baseline_modified_song/pycaret/best_model')


# Feature Imortance : Feature 중요도 확인(autoML)
# plot feature importance


# plot_model(best, plot = 'feature') #KNN등은 안될수도잇음


# Prediction
# predict on test set
holdout_pred = exp.predict_model(best, data=test_all)
print(holdout_pred)


# Submission
# 분류 예측 결과 제출 데이터프레임(DataFrame)
# 분류 예측 결과 데이터프레임 파일명을 반드시 clf_submission.csv 로 지정해야합니다.
clf_submission = pd.read_csv("/workspace/Dataset/FSI/sample_submission.csv")
clf_submission["Fraud_Type"] = holdout_pred.prediction_label
clf_submission.head()



# 합성 데이터 생성 결과 제출 데이터프레임(DataFrame)
# 합성 데이터 생성 결과 데이터프레임 파일명을 반드시 syn_submission.csv 로 지정해야합니다.
all_synthetic_data.head()
'''
(*) 저장 시 각 파일명을 반드시 확인해주세요.
    1. 분류 예측 결과 데이터프레임 파일명 = clf_submission.csv
    2. 합성 데이터 생성 결과 데이터프레임 파일명 = syn_submission.csv

(*) 제출 파일(zip) 내에 두 개의 데이터프레임이 각각 위의 파일명으로 반드시 존재해야합니다.
(*) 파일명을 일치시키지 않으면 채점이 불가능합니다.
'''

# 폴더 생성 및 작업 디렉토리 변경
os.makedirs('/workspace/Dataset/FSI/clip_downloads/submission', exist_ok=True)
os.chdir("/workspace/Dataset/FSI/clip_downloads/submission")

# CSV 파일로 저장
clf_submission.to_csv('/workspace/Dataset/FSI/clip_downloads/submissionclf_submission.csv', encoding='UTF-8-sig', index=False)
# all_synthetic_data.to_csv('/workspace/Dacon_FSI/baseline_modified_song/pycaret/submission/syn_submission.csv', encoding='UTF-8-sig', index=False)

# ZIP 파일 생성 및 CSV 파일 추가
with zipfile.ZipFile("/workspace/Dataset/FSI/clip_downloads/submission/baseline_submission.zip", 'w') as submission:
    submission.write('clf_submission.csv')
    submission.write('syn_submission.csv')
    
print('Done.')