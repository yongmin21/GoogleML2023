TRAINING_FILE = "./../input/train.csv"
TEST_FILE = "./../input/test.csv"
ORIGINAL_FILE = "./../input/original.csv"
SUBMISSION_FILE = "./../input/sample_submission.csv"

#
CATEGORICAL_FEATURES = ['hospital_number', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time','pain',
'peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','rectal_exam_feces','abdomen', 'lesion_1',
'abdomo_appearance','surgery', 'age', 'surgical_lesion', 'cp_data', 'number_of_treatements']

USELESS_FEATURES = ['id']

TARGET_MAP = {
    'died' : 0,
    'euthanized' : 1,
    'lived' : 2
}

XGB_PARAMS = {'min_child_weight': 5,
    'reg_alpha': 0.014425096788083052,
    'reg_lambda': 0.012345176750382126,
    'gamma': 1,
    'colsample_bytree': 0.5,
    'colsample_bynode': 0.7,
    'colsample_bylevel': 0.7,
    'subsample': 0.95,
    'learning_rate': 0.017,
    'max_depth': 15,
    'max_leaves': 366,
    'verbosity': 0
    }

LGBM_PARAMS = {
    'objective': 'multiclass', 
    'num_class': 3,
    'boosting_type' : 'gbdt',
    'num_leaves': 24,
    'max_depth': 10,
    'n_estimators': 450,
    'learning_rate': 0.08,
    'random_state': 42,
    'verbose': -1,
    'subsample':0.8,
    'colsample_bytree':0.65,
    'reg_alpha':0.0001,
    'reg_lambda':3.5,
    'verbose': -1
    }

HGB_PARAMS = {
    'max_depth' : 4,
    'max_iter' : 80,
    'learning_rate' : 0.1,
    'random_state' : 42,
    'scoring' : 'f1_micro',
    'max_leaf_nodes' : 21,
    'l2_regularization' : 0.1,
    'verbose' : 0
}