import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
#import pyreadr #doesn't work with this R object
import pandas as pd
import os as os
import re as re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pprint import pprint# Look at parameters used by our current forest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
##
# Directory information
os.chdir('path/to/directory/')

main_dir_path = 'path/to/directory/'
master_data_path = main_dir_path+"data/"
##
# Loading data files
meta_fn = master_data_path+"metadata.tab"
meta_df = pd.read_csv(meta_fn,sep="\t")

ab_fn = master_data_path+"ab_titer.tab"
ab_df = pd.read_csv(ab_fn,sep="\t")
ab_df = ab_df.clip(lower=0) # Very cool technique, set lower limit of values to 0

cell_fn = master_data_path+"pbmc_cell_frequency.tab"
cell_df = pd.read_csv(cell_fn,sep="\t")
cell_df = cell_df.clip(lower=0)

ck_fn = master_data_path+"plasma_cytokine_concentration.tab"
ck_df = pd.read_csv(ck_fn,sep="\t")
ck_df = ck_df.clip(lower=0)


##
# Merge into master df
master_df = pd.merge(meta_df,ab_df,how="left",on="specimen_id")
master_df = pd.merge(master_df,cell_df,how="left",on="specimen_id")
master_df = pd.merge(master_df,ck_df,how="left",on="specimen_id")
# Calculate age from year_of_birth
master_df['age'] = master_df["year_of_birth"].apply(lambda x: x.split("-")[0])
master_df["age"] = master_df["age"].apply(lambda x: int(x))
master_df["age"] = master_df["age"].apply(lambda x: 2020 - x)

# Calculate average of PT for each specimen
master_df["IgG_PT_sum"] = master_df[['IgG_PT','IgG1_PT','IgG2_PT','IgG3_PT','IgG4_PT']].sum(axis=1)
master_df["IgG_PRN_sum"] = master_df[['IgG_PRN','IgG1_PRN','IgG2_PRN','IgG3_PRN','IgG4_PRN']].sum(axis=1)
master_df["IgG_FHA_sum"] = master_df[['IgG_FHA','IgG1_FHA','IgG2_FHA','IgG3_FHA','IgG4_FHA']].sum(axis=1)
master_df["IgG_DT_sum"] = master_df[['IgG1_DT','IgG2_DT','IgG3_DT','IgG4_DT']].sum(axis=1)
master_df["IgG_OVA_sum"] = master_df[['IgG1_OVA','IgG2_OVA','IgG3_OVA','IgG4_OVA']].sum(axis=1)

# one-hot encoding vaccine type in master_df
master_df["wP_onehot"] = master_df["infancy_vac"].apply(lambda x: 1 if x == "wP" else 0)
master_df["aP_onehot"] = master_df["infancy_vac"].apply(lambda x: 1 if x == "aP" else 0)
# one-hot encoding sex
master_df["sex_M_onehot"] = master_df["biological_sex"].apply(lambda x: 1 if x == "Male" else 0)
master_df["sex_F_onehot"] = master_df["biological_sex"].apply(lambda x: 1 if x == "Female" else 0)

# Modify master_df so that the value of 0 is set to the median value
# Below command takes out all samples where IgG_PT is 0 then sets to median
# training_df[~training_df["subject_id"].isin([75,84,86,89,92])]["IgG_PT"].median() -> 1.11679270812424
master_df["IgG_PT"] = master_df["IgG_PT"].apply(lambda x: x if x > 0 else 1.11679270812424)

# Remove subjects 2, 8, 82, 87, 88 from master as determined by looking at pre_df and d14_df for IgG_PT measurements
master_df = master_df[~master_df["subject_id"].isin([2,8,82,87,88])]

# subset to only pre-boost & make new df that's only post boost day 14
pre_df = master_df[master_df["planned_day_relative_to_boost"] < 1]
pre_specimen_list = list(pre_df["specimen_id"])

d14_df = master_df[master_df["planned_day_relative_to_boost"] == 14]
# Rank the d14 IgG avg value
tmp = d14_df.copy()
tmp['rank'] = d14_df['IgG_PT'].rank(ascending=False).astype(int)
d14_df = tmp

# Find fold-change rank
fold_change_df = d14_df[["subject_id","IgG_PT"]]
fold_change_df = fold_change_df.rename(columns={'IgG_PT':"IgG_PT_d14"})
tmp_df = pre_df[["subject_id","IgG_PT"]]
fold_change_df = pd.merge(fold_change_df,tmp_df, how="inner", on="subject_id")
fold_change_df["fold_change"] = fold_change_df["IgG_PT_d14"] / fold_change_df["IgG_PT"]

# Samples where fold_change is weird
print("Samples with NA fold change",fold_change_df[fold_change_df["fold_change"].isna()])
# Drop subjects 2, 8, 82, 87, 88 at the end
fold_change_df = fold_change_df[~fold_change_df["fold_change"].isna()]


# Make final rank df from fold_change_df
tmp = fold_change_df.copy()
tmp['rank'] = fold_change_df['IgG_PT_d14'].rank(ascending=False).astype(int)
tmp['rank_fold_change'] = fold_change_df['fold_change'].rank(ascending=False).astype(int)
fold_change_df = tmp


# Add rank data to master df
fold_change_merge_df = fold_change_df[['subject_id',"rank","rank_fold_change"]]
master_df = pd.merge(master_df,fold_change_merge_df,on="subject_id",how="left")
# Drop any more samples that have NaN in rank
master_df = master_df[~master_df["rank"].isna()]
# Add column for imputing cell counts
master_df['imputed'] = (~master_df['Monocytes'].isna()).astype(int)
master_df.to_excel("master_df.xlsx",index=False)

# Prepare test training data for scikit learn by removing a sample and making a column that evaluates what that samples rank is
# Do this in a function so it can be iterated over
# Subset master to day 0
# drop appropriate columns not used for training
training_df = master_df[master_df["planned_day_relative_to_boost"] < 1]
cols_to_drop = ['specimen_id', 'actual_day_relative_to_boost', 'planned_day_relative_to_boost', 'specimen_type', 'visit', 'infancy_vac', 'biological_sex', 'ethnicity', 'race', 'year_of_birth', 'date_of_boost', 'dataset', 'timepoint','rank', 'rank_fold_change']
actual_cols = ['subject_id','rank','rank_fold_change']
actual_df = training_df[actual_cols]
training_df = training_df.drop(columns=cols_to_drop)
subject_list=list(actual_df["subject_id"])
actual_df.to_excel("actual_df.xlsx",index=False)
training_df.to_excel("training_df.xlsx",index=False)

##
# Loading data files for master testing set
master_testing_data_path = main_dir_path+"analysis/prediction_data/"
meta_test_fn = master_testing_data_path+"subject_specimen_pred.tab"
meta_test_df = pd.read_csv(meta_test_fn,sep="\t")

ab_test_fn = master_testing_data_path+"ab_titer_pred.tab"
ab_test_df = pd.read_csv(ab_test_fn,sep="\t")
ab_test_df = ab_test_df.clip(lower=0) # Very cool technique, set lower limit of values to 0

cell_test_fn = master_testing_data_path+"pbmc_cell_frequency_pred.tab"
cell_test_df = pd.read_csv(cell_test_fn,sep="\t")
cell_test_df = cell_test_df.clip(lower=0)

ck_test_fn = master_testing_data_path+"plasma_cytokine_concentration_pred.tab"
ck_test_df = pd.read_csv(ck_test_fn,sep="\t")
ck_test_df = ck_test_df.clip(lower=0)


##
# Merge testing data into master testing df
master_test_df = pd.merge(meta_test_df,ab_test_df,how="left",on="specimen_id")
master_test_df = pd.merge(master_test_df,cell_test_df,how="left",on="specimen_id")
master_test_df = pd.merge(master_test_df,ck_test_df,how="left",on="specimen_id")
# Calculate age from year_of_birth
master_test_df['age'] = master_test_df["year_of_birth"].apply(lambda x: x.split("-")[0])
master_test_df["age"] = master_test_df["age"].apply(lambda x: int(x))
master_test_df["age"] = master_test_df["age"].apply(lambda x: 2020 - x)

# one-hot encoding vaccine type in master_test_df
master_test_df["wP_onehot"] = master_test_df["infancy_vac"].apply(lambda x: 1 if x == "wP" else 0)
master_test_df["aP_onehot"] = master_test_df["infancy_vac"].apply(lambda x: 1 if x == "aP" else 0)
# one-hot encoding sex
master_test_df["sex_M_onehot"] = master_test_df["biological_sex"].apply(lambda x: 1 if x == "Male" else 0)
master_test_df["sex_F_onehot"] = master_test_df["biological_sex"].apply(lambda x: 1 if x == "Female" else 0)

# Calculate average of PT for each specimen
master_test_df["IgG_PT_sum"] = master_test_df[['IgG_PT','IgG1_PT','IgG2_PT','IgG3_PT','IgG4_PT']].sum(axis=1)
master_test_df["IgG_PRN_sum"] = master_test_df[['IgG_PRN','IgG1_PRN','IgG2_PRN','IgG3_PRN','IgG4_PRN']].sum(axis=1)
master_test_df["IgG_FHA_sum"] = master_test_df[['IgG_FHA','IgG1_FHA','IgG2_FHA','IgG3_FHA','IgG4_FHA']].sum(axis=1)
master_test_df["IgG_DT_sum"] = master_test_df[['IgG1_DT','IgG2_DT','IgG3_DT','IgG4_DT']].sum(axis=1)
master_test_df["IgG_OVA_sum"] = master_test_df[['IgG1_OVA','IgG2_OVA','IgG3_OVA','IgG4_OVA']].sum(axis=1)

# Make imputed column
master_test_df['imputed'] = (~master_test_df['Monocytes'].isna()).astype(int)
master_test_df.to_excel("master_test_df.xlsx",index=False)
##
master_test_df = pd.read_excel("master_test_df.xlsx")
test_input_df = master_test_df[master_test_df["planned_day_relative_to_boost"]==0]
# need to drop columns not in the trianing set master_df
cols_to_drop = ['ASCs (Plasmablasts)', 'CD3CD19']
test_input_df = test_input_df.drop(cols_to_drop,axis=1)
# now drop rest of the cols to make it congruent with training set
cols_to_drop = ['specimen_id', 'actual_day_relative_to_boost', 'planned_day_relative_to_boost', 'specimen_type', 'visit', 'infancy_vac', 'biological_sex', 'ethnicity', 'race', 'year_of_birth', 'date_of_boost', 'dataset', 'timepoint']
test_input_df = test_input_df.drop(cols_to_drop,axis=1)
test_input_df.to_excel("test_input_df.xlsx",index=False)

# Test cols are equal
test_col = set(list(test_input_df.columns))
train_col = set(list(training_df.columns))
test_unique_col = test_col - train_col
train_unique_col = train_col - test_col
print(f"Unique train cols {train_unique_col} Unique test cols {test_unique_col}")
# Append dataframes for PCA testing
train_df_tmp = training_df.copy()
train_df_tmp["dataset"] = "Train"
test_df_tmp = test_input_df.copy()
test_df_tmp["dataset"] = "Test"
combined_train_test_df = train_df_tmp.append(test_df_tmp,ignore_index=True)
##
# Just to load in previously generated datasets that have been created above
master_df = pd.read_excel("master_df.xlsx")
training_df = pd.read_excel("training_df.xlsx")
actual_df = pd.read_excel("actual_df.xlsx")
test_input_df = pd.read_excel("test_input_df.xlsx")
subject_list=list(training_df["subject_id"])
##
# Define IgG rank training function
# Takes in the training and actual rank dfs and finds an optimal settings match using randomized parameter training.
# Outputs dictionary with optimized random forest and following information
# dictionary keys = "model_" + subject_id
# dictionary values = [<model> , accuracy , X_train, X_test, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed , model_12_df, rf_params, feature_importances]
def train_models_rank_optimization(subject_id, training_df, actual_df, test_size=0.2):
    model_ran_seed = random.randint(1,10000)
    #model_ran_seed = 42 # for testing
    rank_values = actual_df.loc[actual_df['subject_id'] == subject_id, 'rank']
    if not rank_values.empty:
        rank_value = rank_values.iloc[0]
        rank_fold_change_values = actual_df.loc[actual_df['subject_id'] == subject_id, 'rank_fold_change']
        rank_fold_change_value = rank_fold_change_values.iloc[0]
    else:
        return print("Subject ID %i not found" % subject_id)

    model_12_df = training_df[training_df["subject_id"] != subject_id]
    model_12_df = pd.merge(model_12_df, actual_df, how="left", on="subject_id")
    model_12_df["actual_rank"] = model_12_df["rank"].apply(lambda x: 1 if x > rank_value else 0)
    model_12_df["actual_rank_fold_change"] = model_12_df["rank_fold_change"].apply(lambda x: 1 if x > rank_fold_change_value else 0)
    rank_actual = model_12_df["actual_rank"]
    rank_fold_change_actual = model_12_df["actual_rank_fold_change"]
    X = model_12_df.drop(['subject_id', 'actual_rank', "actual_rank_fold_change", "rank", "rank_fold_change"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, rank_actual, test_size=test_size, random_state=model_ran_seed)
    imputer = SimpleImputer(strategy='median')  # or mean, most_frequent
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # modifying random forest parameters to see if anything improves
    #clf = RandomForestClassifier(random_state=model_ran_seed)
    #clf = RandomForestClassifier(random_state=model_ran_seed, n_estimators=201)
    random_grid = {'bootstrap': [True, False],
 'max_depth': [20, 30, 40, 50, 60, 70, 80, None],
 'max_features': ['log2', 'sqrt'],
 'min_samples_leaf': [1, 2, 4, 6, 8],
 'min_samples_split': [2, 5, 10, 15, 20],
 'n_estimators': [201, 401, 601, 801, 1001]}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                   random_state=model_ran_seed, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train_imputed, y_train)
    rf_params = rf_random.best_params_
    #rf_random.fit(X_train_imputed, y_train)
    clf = RandomForestClassifier(random_state=model_ran_seed)
    clf.set_params(**rf_params)
    clf.fit(X_train_imputed, y_train)
    y_pred = clf.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred)

    feature_names = list(X.columns)
    importances = clf.feature_importances_
    feature_importances = list(zip(feature_names, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return [clf , accuracy , X_train_imputed, X_test_imputed, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed, model_12_df, rf_params, feature_importances]
##
# Initialize dictionary and run model trianing loop on all samples
model_dict={}
for i in subject_list:
    model_dict[f"model_{i}"] = 0

training_df = training_df.reset_index(drop=True)
# Confirm feature length
rank_columns = [col for col in training_df.columns if 'rank' in col]
for subject_id in subject_list[:]:
    #model_dict[f"model_{subject_id}"] = train_models_rank(subject_id,training_df,actual_df,0.2)
    model_dict[f"model_{subject_id}"] = train_models_rank_optimization(subject_id, training_df, actual_df, 0.2)
    clf = model_dict[f"model_{subject_id}"][0]
    accuracy = model_dict[f"model_{subject_id}"][1]
    X_test_imputed = model_dict[f"model_{subject_id}"][3]
    y_test=model_dict[f"model_{subject_id}"][5]
    model_rank_value = model_dict[f"model_{subject_id}"][6]
    model_rank_fc_value = model_dict[f"model_{subject_id}"][7]
    model_df = model_dict[f"model_{subject_id}"][9]
    print(f"---- ID {subject_id} ----  Rank {model_rank_value} ---- Acc {accuracy}")
    for x in range(len(X_test_imputed)):
        new_x = X_test_imputed[x]
        new_x_rs = new_x.reshape(1,-1) # Reshape like this when it's just a single sample
        new_pred = clf.predict(new_x_rs) # Gives array with prediction
        new_probability = clf.predict_proba(new_x_rs) # Gives array with probability it's in class 0 vs class 1
        true_prediction= new_pred[0] == y_test.reset_index(drop=True)[x]
        true_rank_index = y_test.reset_index()
        true_rank_index = true_rank_index.loc[x,"index"]
        true_subject_id = model_df.loc[true_rank_index,["subject_id","rank"]][0]
        true_rank = model_df.loc[true_rank_index, ["subject_id", "rank"]][1]
        #print(f"Input {x} | Prediction {new_pred} | Probability {new_probability} | {true_prediction} | True rank {true_rank} | Subject id {true_subject_id}")

# Save as independent variable
optimized_rank_dict = model_dict
##
# This takes in an input array of features to input into the trained forest model, the dictionary with each model, the output feature dictionary, and a factor to scale the probability
# This outputs in the XG_feature_dictionary a value for each model which is a prediction (-1 or 1) times (probability ** probability factor)
def forest_quorum_model(input_ar , XG_feature_dict , model_dict, probability_factor=3):
    # input_ar is expected to be an array that is [[sample_id,feature_1,...]]
    subject_id = input_ar[0]
    input_ar = input_ar[1:]
    input_ar = input_ar.reshape(1,-1)
    XG_feature_dict["subject_id"].append(subject_id)
    for model in model_dict:
        clf = model_dict[model][0]
        prediction = clf.predict(input_ar)
        prediction = prediction[0]
        probability = clf.predict_proba(input_ar)
        probability = [1 if len(probability[0])==1 else probability[0][prediction]][0]
        #print(probability)
        #probability = probability[0][prediction]
        out_feature = [-1 if prediction == 0 else prediction][0] # This changes 0 to -1 so higher than = -1 lower than = 1 for every model
        #print(f"outfeat {out_feature} Pred {prediction} Probability {probability}")
        out_feature = out_feature*(probability**probability_factor) # This scales the probability impact, set to 0 to ignore probability
        XG_feature_dict[model].append(out_feature)
    return XG_feature_dict
##
# Create starting dictionary
XG_feature_dict = {}
XG_feature_dict["subject_id"] = []
for model in optimized_rank_dict.keys():
    XG_feature_dict[model] = []
# impute missing values using median
imputer = SimpleImputer(strategy='median')
train_imp_array = imputer.fit_transform(training_df)
# For each subject, generate a prediction from each model to create a subject_id x model_id matrix of predictions
for input_ar in train_imp_array:
    XG_feature_dict = forest_quorum_model(input_ar, XG_feature_dict, optimized_rank_dict, 3)

##
XG_feature_df = pd.DataFrame(XG_feature_dict)
XG_feature_df_actual = pd.merge(XG_feature_df,actual_df,on="subject_id",how="left")

##
# First subset dataset to training/test
XG_rank_df = XG_feature_df_actual.drop(["rank_fold_change"],axis=1)
gss = GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 7).split(XG_rank_df, groups=XG_rank_df['subject_id'])

X_train_inds, X_test_inds = next(gss)
XG_train = XG_rank_df.iloc[X_train_inds]
X_train = XG_train.loc[:, ~XG_train.columns.isin(['subject_id','rank'])]
y_train = XG_train.loc[:, XG_train.columns.isin(['rank'])]

XG_test = XG_rank_df.iloc[X_test_inds]
X_test = XG_test.loc[:, ~XG_test.columns.isin(['subject_id','rank'])]
y_test = XG_test.loc[:, XG_test.columns.isin(['rank'])]
##

dtrain = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'reg:squarederror',  # or another ranking objective
    'random_state' : 42,
    'max_depth' : 6, # default 3
    'learning_rate': 0.1,
    'eval_metric': 'ndcg',  # or another suitable metric for ranking
    # other parameters as needed
}

bst = xgb.train(params, dtrain, num_boost_round=100)
##

dtest = xgb.DMatrix(X_test, label=y_test)
y_pred = bst.predict(dtest)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
##
# Calculate spearmans
output_real_predict_df = y_test.copy()
output_real_predict_df["y_pred"] = y_pred
# Add 1-X ranks to the y_pred and rank columns
output_real_predict_df['rank_comparison'] = output_real_predict_df['rank'].rank(ascending=False).astype(int)
output_real_predict_df['y_pred_comparison'] = output_real_predict_df['y_pred'].rank(ascending=False).astype(int)
spearman_correlation = output_real_predict_df[['rank_comparison', 'y_pred_comparison']].corr(method='spearman').iloc[0, 1]

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² score: {r2}")
print(f"Spearman Correlation: {spearman_correlation}")
##
# Run actual test dataframe through rank model to determine final
# test_input_df -> the test data formatted for input into the model

# Create starting dictionary
XG_feature_predict_dict = {}
XG_feature_predict_dict["subject_id"] = []
for model in optimized_rank_dict.keys():
    XG_feature_predict_dict[model] = []

imputer = SimpleImputer(strategy='median')
test_imp_array = imputer.fit_transform(test_input_df)


for input_ar in test_imp_array:
    XG_feature_predict_dict = forest_quorum_model(input_ar, XG_feature_predict_dict, optimized_rank_dict, 3)


##
XG_feature_predict_df = pd.DataFrame(XG_feature_predict_dict)
XG_feature_predict_subject_id = XG_feature_predict_df["subject_id"].to_numpy()
XG_predict_id_df = XG_feature_predict_df["subject_id"].reset_index() # Gets DF with index and subject_id columns
XG_predict_input_df = XG_feature_predict_df.loc[:, ~XG_feature_predict_df.columns.isin(['subject_id'])] #Keep subject ID in there
d_predict = xgb.DMatrix(XG_predict_input_df)
prediction_final = bst.predict(d_predict)
prediction_dict = {"IgG_rank_prediction": prediction_final.tolist()}
prediction_dict["subject_id"] = XG_feature_predict_subject_id.tolist()
prediction_final_df = pd.DataFrame(prediction_dict)
prediction_final_df["IgG_rank"] = prediction_final_df["IgG_rank_prediction"].rank(ascending=True).astype(int)
prediction_final_df.to_excel("IGG_rank_prediction.xlsx",index=False)
#
#
# End of IgG absolute Rank prediction
#
#
#
# The IgG fold change rank prediction uses the same general model structure, retrained on fold change
#
##
def train_models_rank_fold_optimization(subject_id, training_df, actual_df, test_size=0.2):
    model_ran_seed = random.randint(1,10000)
    #model_ran_seed = 42 # for testing
    rank_values = actual_df.loc[actual_df['subject_id'] == subject_id, 'rank']
    if not rank_values.empty:
        rank_value = rank_values.iloc[0]
        rank_fold_change_values = actual_df.loc[actual_df['subject_id'] == subject_id, 'rank_fold_change']
        rank_fold_change_value = rank_fold_change_values.iloc[0]
    else:
        return print("Subject ID %i not found" % subject_id)

    model_12_df = training_df[training_df["subject_id"] != subject_id]
    model_12_df = pd.merge(model_12_df, actual_df, how="left", on="subject_id")
    model_12_df["actual_rank"] = model_12_df["rank"].apply(lambda x: 1 if x > rank_value else 0)
    model_12_df["actual_rank_fold_change"] = model_12_df["rank_fold_change"].apply(lambda x: 1 if x > rank_fold_change_value else 0)
    rank_actual = model_12_df["actual_rank"]
    rank_fold_change_actual = model_12_df["actual_rank_fold_change"]
    X = model_12_df.drop(['subject_id', 'actual_rank', "actual_rank_fold_change", "rank", "rank_fold_change"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, rank_fold_change_actual, test_size=test_size, random_state=model_ran_seed)
    imputer = SimpleImputer(strategy='median')  # or mean, most_frequent
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # modifying random forest parameters to see if anything improves
    #clf = RandomForestClassifier(random_state=model_ran_seed)
    #clf = RandomForestClassifier(random_state=model_ran_seed, n_estimators=201)
    random_grid = {'bootstrap': [True, False],
 'max_depth': [20, 30, 40, 50, 60, 70, 80, None],
 'max_features': ['log2', 'sqrt'],
 'min_samples_leaf': [1, 2, 4, 6, 8],
 'min_samples_split': [2, 5, 10, 15, 20],
 'n_estimators': [201, 401, 601, 801, 1001]}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                   random_state=model_ran_seed, n_jobs=-1)  # Fit the random search model

    rf_random.fit(X_train_imputed, y_train)
    rf_params = rf_random.best_params_
    #rf_random.fit(X_train_imputed, y_train)
    clf = RandomForestClassifier(random_state=model_ran_seed)
    clf.set_params(**rf_params)
    clf.fit(X_train_imputed, y_train)
    y_pred = clf.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred)

    feature_names = list(X.columns)
    importances = clf.feature_importances_
    feature_importances = list(zip(feature_names, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return [clf , accuracy , X_train_imputed, X_test_imputed, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed, model_12_df, rf_params, feature_importances]


##
#
# Model Training loop for rank FC
#
# Testing train_model_rank
# <model> , accuracy , X_train, X_test, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed , model_12_df
subject_list=list(actual_df["subject_id"])
optimized_fc_dict={}
for i in subject_list:
    optimized_fc_dict[f"model_{i}"] = 0

training_df = training_df.reset_index(drop=True)
# Confirm feature length
rank_columns = [col for col in training_df.columns if 'rank' in col]
for subject_id in subject_list[:]:
    optimized_fc_dict[f"model_{subject_id}"] = train_models_rank_fold_optimization(subject_id,training_df,actual_df,0.2)
    clf = optimized_fc_dict[f"model_{subject_id}"][0]
    accuracy = optimized_fc_dict[f"model_{subject_id}"][1]
    X_test_imputed = optimized_fc_dict[f"model_{subject_id}"][3]
    y_test=optimized_fc_dict[f"model_{subject_id}"][5]
    model_rank_value = optimized_fc_dict[f"model_{subject_id}"][6]
    model_rank_fc_value = optimized_fc_dict[f"model_{subject_id}"][7]
    model_df = optimized_fc_dict[f"model_{subject_id}"][9]
    print(f"---- ID {subject_id} ----  Rank fc {model_rank_fc_value} ---- Acc {accuracy}")
    for x in range(len(X_test_imputed)):
        new_x = X_test_imputed[x]
        new_x_rs = new_x.reshape(1,-1) # Reshape like this when it's just a single sample
        new_pred = clf.predict(new_x_rs) # Gives array with prediction
        new_probability = clf.predict_proba(new_x_rs) # Gives array with probability it's in class 0 vs class 1
        true_prediction= new_pred[0] == y_test.reset_index(drop=True)[x]
        true_rank_index = y_test.reset_index()
        true_rank_index = true_rank_index.loc[x,"index"]
        true_subject_id = model_df.loc[true_rank_index,["subject_id","rank"]][0]
        true_rank = model_df.loc[true_rank_index, ["subject_id", "rank"]][1]
        #print(f"Input {x} | Prediction {new_pred} | Probability {new_probability} | {true_prediction} | True rank {true_rank} | Subject id {true_subject_id}")
##
# Create starting dictionary
XG_feature_dict = {}
XG_feature_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_dict[model] = []

imputer = SimpleImputer(strategy='median')
train_imp_array = imputer.fit_transform(training_df)


for input_ar in train_imp_array:
    XG_feature_fc_dict = forest_quorum_model(input_ar, XG_feature_dict, optimized_fc_dict, 3)
##
XG_feature_df = pd.DataFrame(XG_feature_fc_dict)
XG_feature_df_actual = pd.merge(XG_feature_df,actual_df,on="subject_id",how="left")

##
# We got it! time for XGboost!
# First subset dataset
XG_rank_fc_df = XG_feature_df_actual.drop(["rank"],axis=1)
gss = GroupShuffleSplit(test_size=.2, n_splits=1, random_state = 7).split(XG_rank_fc_df, groups=XG_rank_fc_df['subject_id'])


X_train_inds, X_test_inds = next(gss)
XG_train = XG_rank_fc_df.iloc[X_train_inds]
X_train = XG_train.loc[:, ~XG_train.columns.isin(['subject_id','rank_fold_change'])]
y_train = XG_train.loc[:, XG_train.columns.isin(['rank_fold_change'])]

XG_test = XG_rank_fc_df.iloc[X_test_inds]
X_test = XG_test.loc[:, ~XG_test.columns.isin(['subject_id','rank_fold_change'])]
y_test = XG_test.loc[:, XG_test.columns.isin(['rank_fold_change'])]
##

dtrain_fold = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'reg:squarederror',  # or another ranking objective
    'random_state' : 42,
    'max_depth' : 6, # default 3
    'learning_rate': 0.1,
    'eval_metric': 'ndcg',  # or another suitable metric for ranking
    # other parameters as needed
}

bst_fold = xgb.train(params, dtrain_fold, num_boost_round=100)
##

dtest_fold = xgb.DMatrix(X_test, label=y_test)
y_pred = bst_fold.predict(dtest_fold)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
##
# Calculate spearmans
output_real_predict_df = y_test.copy()
output_real_predict_df["y_pred"] = y_pred
# Add 1-X ranks to the y_pred and rank columns
output_real_predict_df['rank_comparison'] = output_real_predict_df['rank_fold_change'].rank(ascending=False).astype(int)
output_real_predict_df['y_pred_comparison'] = output_real_predict_df['y_pred'].rank(ascending=False).astype(int)
spearman_correlation = output_real_predict_df[['rank_comparison', 'y_pred_comparison']].corr(method='spearman').iloc[0, 1]

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² score: {r2}")
print(f"Spearman Correlation: {spearman_correlation}")
##
# Saving optimized model dictionary as new variable for testing
#optimized_fc_dict = model_dict

##

# Create starting dictionary for prediction
XG_feature_predict_dict = {}
XG_feature_predict_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_predict_dict[model] = []

imputer = SimpleImputer(strategy='median')
test_imp_array = imputer.fit_transform(test_input_df)


for input_ar in test_imp_array:
    XG_feature_predict_dict = forest_quorum_model(input_ar, XG_feature_predict_dict, optimized_fc_dict, 3)


##
XG_feature_predict_df = pd.DataFrame(XG_feature_predict_dict)
XG_feature_predict_subject_id = XG_feature_predict_df["subject_id"].to_numpy()
XG_predict_id_df = XG_feature_predict_df["subject_id"].reset_index() # Gets DF with index and subject_id columns
XG_predict_input_df = XG_feature_predict_df.loc[:, ~XG_feature_predict_df.columns.isin(['subject_id'])] #Keep subject ID in there
d_predict = xgb.DMatrix(XG_predict_input_df)
prediction_final = bst_fold.predict(d_predict)
prediction_dict = {"IgG_rank_fold_chage_prediction": prediction_final.tolist()}
prediction_dict["subject_id"] = XG_feature_predict_subject_id.tolist()
prediction_final_df = pd.DataFrame(prediction_dict)
prediction_final_df["IgG_rank_fc"] = prediction_final_df["IgG_rank_fold_chage_prediction"].rank(ascending=True).astype(int)
prediction_final_df.to_excel("IGG_rank_fold_change_prediction.xlsx",index=False)
##
#
# End rank fold change IgG prediction
#
#
##
#
#
# Prep the day 1 monocyte Dataset
#
#
#
# training_df = monocyte_training_df
# master_df = monocyte_df
# actual_df = monocyte_actual_df
# rank = Monocyte_rank | rank_fold_change = Monocyte_fold_change


pre_df = master_df[master_df["planned_day_relative_to_boost"] < 1]
pre_specimen_list = list(pre_df["specimen_id"])

d1_df = master_df[master_df["planned_day_relative_to_boost"] == 1]
d1_df = d1_df[~d1_df["Monocytes"].isna()]
# Rank the d14 IgG avg value
tmp = d1_df.copy()
tmp['Monocyte_rank'] = d1_df['Monocytes'].rank(ascending=False).astype(int)
d1_df = tmp

# Find fold-change rank
fold_change_df = d1_df[["subject_id","Monocytes"]]
fold_change_df = fold_change_df.rename(columns={'Monocytes':"Monocytes_d1"})
tmp_df = pre_df[["subject_id","Monocytes"]]
fold_change_df = pd.merge(fold_change_df,tmp_df, how="inner", on="subject_id")
fold_change_df["Monocytes_fold_change"] = fold_change_df["Monocytes_d1"] / fold_change_df["Monocytes"]



# Make final rank df from fold_change_df
tmp = fold_change_df.copy()
tmp['Monocyte_rank'] = fold_change_df['Monocytes_d1'].rank(ascending=False).astype(int)
tmp['Monocytes_rank_fc'] = fold_change_df['Monocytes_fold_change'].rank(ascending=False).astype(int)
fold_change_df = tmp


# Add rank data to master df
fold_change_merge_df = fold_change_df[['subject_id','Monocyte_rank',"Monocytes_rank_fc"]]
monocyte_df = pd.merge(master_df,fold_change_merge_df,on="subject_id",how="left")
# Drop samples with no monocytes d1
monocyte_df = monocyte_df[~monocyte_df['Monocyte_rank'].isna()]
monocyte_df = monocyte_df.fillna(monocyte_df.median())

monocyte_df.to_excel("monocyte_df.xlsx",index=False)

# Prepare test training data for scikit learn by removing a sample and making a column that evaluates what that samples rank is
# Do this in a function so it can be iterated over
# Subset master to day 0
# drop appropriate columns not used for training
monocyte_training_df = monocyte_df[monocyte_df["planned_day_relative_to_boost"] < 1]
cols_to_drop = ['specimen_id', 'actual_day_relative_to_boost', 'planned_day_relative_to_boost', 'specimen_type', 'visit', 'infancy_vac', 'biological_sex', 'ethnicity', 'race', 'year_of_birth', 'date_of_boost', 'dataset', 'timepoint','Monocyte_rank', 'Monocytes_rank_fc','rank','rank_fold_change','imputed']
actual_cols = ['subject_id','Monocyte_rank','Monocytes_rank_fc']
monocyte_actual_df = monocyte_training_df[actual_cols]
monocyte_training_df = monocyte_training_df.drop(columns=cols_to_drop)
subject_list=list(monocyte_actual_df["subject_id"])
monocyte_actual_df.to_excel("monocyte_actual_df.xlsx",index=False)
monocyte_training_df.to_excel("monocyte_training_df.xlsx",index=False)
##
# Plots of monocyte actual
# Quick plot rank vs value
plt_name="Monocyte_rank-fold-change_v_fold-change.png"
plt_path = main_dir_path+"analysis/"+plt_name
plt.figure(figsize=(10, 6))
# Creating the plot with seaborn
sns.lineplot(x='Monocytes_rank_fc', y='Monocytes_fold_change', data=fold_change_df) # Figure out what the Y column is called in monocyte fold change df
# Setting Y-axis to logarithmic scale
#plt.yscale('log')
# Adding title and labels
plt.title('Monocyte rank fold change vs fold change value')
plt.xlabel('Monocyte_rank_fold_change')
plt.ylabel('Monocyte_fold_change (log scale)')
# Show the plot
plt.savefig(plt_path,dpi=300)

plt_name="Monocyte_rank_v_monocytes-d1.png"
plt_path = main_dir_path+"analysis/"+plt_name
plt.figure(figsize=(10, 6))
# Creating the plot with seaborn
sns.lineplot(x='Monocyte_rank', y='Monocytes_d1', data=fold_change_df)
# Setting Y-axis to logarithmic scale
#plt.yscale('log')
# Adding title and labels
plt.title('Monocyte rank vs Monocytes value (Day 1)')
plt.xlabel('rank')
plt.ylabel('Monocytes day 1 (log scale)')
# Show the plot
plt.savefig(plt_path,dpi=300)
##

monocyte_df = pd.read_excel("monocyte_df.xlsx")
monocyte_actual_df = pd.read_excel("monocyte_actual_df.xlsx")
monocyte_training_df = pd.read_excel("monocyte_training_df.xlsx")

subject_list=list(monocyte_actual_df["subject_id"])

model_dict={}
for i in subject_list:
    model_dict[f"model_{i}"] = 0
# This creates our dictionary of model variables which will eventually store each model
# Dictionary layout
# <model> , accuracy , X_train, X_test, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed , model_12_df
##
# training_df = monocyte_training_df
# master_df = monocyte_df
# actual_df = monocyte_actual_df
# rank = Monocyte_rank | rank_fold_change = Monocyte_fold_change
# function that takes in subject_id, training_df, and actual_df then outputs the model as a value to the dictionary
def train_monocytes_models_rank_fold_optimization(subject_id, monocyte_training_df, monocyte_actual_df, test_size=0.1):
    model_ran_seed = random.randint(1,10000)
    #model_ran_seed = 42 # for testing
    rank_values = monocyte_actual_df.loc[monocyte_actual_df['subject_id'] == subject_id, 'Monocyte_rank']
    if not rank_values.empty:
        rank_value = rank_values.iloc[0]
        rank_fold_change_values = monocyte_actual_df.loc[monocyte_actual_df['subject_id'] == subject_id, 'Monocytes_rank_fc']
        rank_fold_change_value = rank_fold_change_values.iloc[0]
    else:
        return print("Subject ID %i not found" % subject_id)

    model_12_df = monocyte_training_df[monocyte_training_df["subject_id"] != subject_id]
    model_12_df = pd.merge(model_12_df, monocyte_actual_df, how="left", on="subject_id")
    model_12_df["actual_rank"] = model_12_df['Monocyte_rank'].apply(lambda x: 1 if x > rank_value else 0)
    model_12_df["actual_rank_fold_change"] = model_12_df['Monocytes_rank_fc'].apply(lambda x: 1 if x > rank_fold_change_value else 0)
    rank_actual = model_12_df["actual_rank"]
    rank_fold_change_actual = model_12_df["actual_rank_fold_change"]
    X = model_12_df.drop(['subject_id', 'actual_rank', "actual_rank_fold_change", 'Monocyte_rank', 'Monocytes_rank_fc'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, rank_fold_change_actual, test_size=test_size, random_state=model_ran_seed)
    imputer = SimpleImputer(strategy='median')  # or mean, most_frequent
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # modifying random forest parameters to see if anything improves
    #clf = RandomForestClassifier(random_state=model_ran_seed)
    #clf = RandomForestClassifier(random_state=model_ran_seed, n_estimators=201)
    random_grid = {'bootstrap': [True, False],
 'max_depth': [20, 30, 40, 50, 60, 70, 80, None],
 'max_features': ['log2', 'sqrt'],
 'min_samples_leaf': [1, 2, 4, 6, 8],
 'min_samples_split': [2, 5, 10, 15, 20],
 'n_estimators': [201, 401, 601, 801, 1001]}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                   random_state=model_ran_seed, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train_imputed, y_train)
    rf_params = rf_random.best_params_
    #rf_random.fit(X_train_imputed, y_train)
    clf = RandomForestClassifier(random_state=model_ran_seed)
    clf.set_params(**rf_params)
    clf.fit(X_train_imputed, y_train)
    y_pred = clf.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred)

    feature_names = list(X.columns)
    importances = clf.feature_importances_
    feature_importances = list(zip(feature_names, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return [clf , accuracy , X_train_imputed, X_test_imputed, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed, model_12_df, rf_params, feature_importances]


def train_monocytes_models_rank_optimization(subject_id, monocyte_training_df, monocyte_actual_df, test_size=0.1):
    model_ran_seed = random.randint(1,10000)
    #model_ran_seed = 42 # for testing
    rank_values = monocyte_actual_df.loc[monocyte_actual_df['subject_id'] == subject_id, 'Monocyte_rank']
    if not rank_values.empty:
        rank_value = rank_values.iloc[0]
        rank_fold_change_values = monocyte_actual_df.loc[monocyte_actual_df['subject_id'] == subject_id, 'Monocytes_rank_fc']
        rank_fold_change_value = rank_fold_change_values.iloc[0]
    else:
        return print("Subject ID %i not found" % subject_id)

    model_12_df = monocyte_training_df[monocyte_training_df["subject_id"] != subject_id]
    model_12_df = pd.merge(model_12_df, monocyte_actual_df, how="left", on="subject_id")
    model_12_df["actual_rank"] = model_12_df['Monocyte_rank'].apply(lambda x: 1 if x > rank_value else 0)
    model_12_df["actual_rank_fold_change"] = model_12_df['Monocytes_rank_fc'].apply(lambda x: 1 if x > rank_fold_change_value else 0)
    rank_actual = model_12_df["actual_rank"]
    rank_fold_change_actual = model_12_df["actual_rank_fold_change"]
    X = model_12_df.drop(['subject_id', 'actual_rank', "actual_rank_fold_change", 'Monocyte_rank', 'Monocytes_rank_fc'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, rank_actual, test_size=test_size, random_state=model_ran_seed)
    imputer = SimpleImputer(strategy='median')  # or mean, most_frequent
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    # modifying random forest parameters to see if anything improves
    #clf = RandomForestClassifier(random_state=model_ran_seed)
    #clf = RandomForestClassifier(random_state=model_ran_seed, n_estimators=201)
    random_grid = {'bootstrap': [True, False],
 'max_depth': [20, 30, 40, 50, 60, 70, 80, None],
 'max_features': ['log2', 'sqrt'],
 'min_samples_leaf': [1, 2, 4, 6, 8],
 'min_samples_split': [2, 5, 10, 15, 20],
 'n_estimators': [201, 401, 601, 801, 1001]}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                   random_state=model_ran_seed, n_jobs=-1)  # Fit the random search model
    rf_random.fit(X_train_imputed, y_train)
    rf_params = rf_random.best_params_
    #rf_random.fit(X_train_imputed, y_train)
    clf = RandomForestClassifier(random_state=model_ran_seed)
    clf.set_params(**rf_params)
    clf.fit(X_train_imputed, y_train)
    y_pred = clf.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, y_pred)

    feature_names = list(X.columns)
    importances = clf.feature_importances_
    feature_importances = list(zip(feature_names, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    return [clf , accuracy , X_train_imputed, X_test_imputed, y_train, y_test, rank_value, rank_fold_change_value, model_ran_seed, model_12_df, rf_params, feature_importances]


def forest_quorum_model(input_ar , XG_feature_dict , model_dict, probability_factor=3):
    # input_ar is expected to be an array that is [[sample_id,feature_1,...]]
    subject_id = input_ar[0]
    input_ar = input_ar[1:]
    input_ar = input_ar.reshape(1,-1)
    XG_feature_dict["subject_id"].append(subject_id)
    for model in model_dict:
        clf = model_dict[model][0]
        prediction = clf.predict(input_ar)
        prediction = prediction[0]
        probability = clf.predict_proba(input_ar)
        probability = [1 if len(probability[0])==1 else probability[0][prediction]][0]
        #print(probability)
        #probability = probability[0][prediction]
        out_feature = [-1 if prediction == 0 else prediction][0] # This changes 0 to -1 so higher than = -1 lower than = 1 for every model
        #print(f"outfeat {out_feature} Pred {prediction} Probability {probability}")
        out_feature = out_feature*(probability**probability_factor) # This scales the probability impact, set to 0 to ignore probability
        XG_feature_dict[model].append(out_feature)
    return XG_feature_dict
##
#
# Monocytes rank training
#
optimized_fc_dict={}
for i in subject_list:
    optimized_fc_dict[f"model_{i}"] = 0
monocyte_training_df = monocyte_training_df.reset_index(drop=True)
# Confirm feature length
rank_columns = [col for col in monocyte_training_df.columns if 'Monocyte_rank' in col]
for subject_id in subject_list[:]:
    optimized_fc_dict[f"model_{subject_id}"] = train_monocytes_models_rank_optimization(subject_id,monocyte_training_df,monocyte_actual_df,0.02)
    clf = optimized_fc_dict[f"model_{subject_id}"][0]
    accuracy = optimized_fc_dict[f"model_{subject_id}"][1]
    X_test_imputed = optimized_fc_dict[f"model_{subject_id}"][3]
    y_test=optimized_fc_dict[f"model_{subject_id}"][5]
    model_rank_value = optimized_fc_dict[f"model_{subject_id}"][6]
    model_rank_fc_value = optimized_fc_dict[f"model_{subject_id}"][7]
    model_df = optimized_fc_dict[f"model_{subject_id}"][9]
    print(f"---- ID {subject_id} ----  Rank fc {model_rank_fc_value} ---- Acc {accuracy}")
    for x in range(len(X_test_imputed)):
        new_x = X_test_imputed[x]
        new_x_rs = new_x.reshape(1,-1) # Reshape like this when it's just a single sample
        new_pred = clf.predict(new_x_rs) # Gives array with prediction
        new_probability = clf.predict_proba(new_x_rs) # Gives array with probability it's in class 0 vs class 1
        true_prediction= new_pred[0] == y_test.reset_index(drop=True)[x]
        true_rank_index = y_test.reset_index()
        true_rank_index = true_rank_index.loc[x,"index"]
        true_subject_id = model_df.loc[true_rank_index,["subject_id",'Monocyte_rank']][0]
        true_rank = model_df.loc[true_rank_index, ["subject_id", 'Monocyte_rank']][1]
        #print(f"Input {x} | Prediction {new_pred} | Probability {new_probability} | {true_prediction} | True rank {true_rank} | Subject id {true_subject_id}")
##
# Create starting dictionary
XG_feature_dict = {}
XG_feature_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_dict[model] = []
imputer = SimpleImputer(strategy='median')
train_imp_array = imputer.fit_transform(monocyte_training_df)
for input_ar in train_imp_array:
    XG_feature_fc_dict = forest_quorum_model(input_ar, XG_feature_dict, optimized_fc_dict, 3)
XG_feature_df = pd.DataFrame(XG_feature_fc_dict)
XG_feature_df_actual = pd.merge(XG_feature_df,monocyte_actual_df,on="subject_id",how="left")
XG_rank_fc_df = XG_feature_df_actual.drop(['Monocytes_rank_fc'],axis=1)
gss = GroupShuffleSplit(test_size=.02, n_splits=1, random_state = 7).split(XG_rank_fc_df, groups=XG_rank_fc_df['subject_id'])
X_train_inds, X_test_inds = next(gss)
XG_train = XG_rank_fc_df.iloc[X_train_inds]
X_train = XG_train.loc[:, ~XG_train.columns.isin(['subject_id','Monocyte_rank'])]
y_train = XG_train.loc[:, XG_train.columns.isin(['Monocyte_rank'])]
XG_test = XG_rank_fc_df.iloc[X_test_inds]
X_test = XG_test.loc[:, ~XG_test.columns.isin(['subject_id','Monocyte_rank'])]
y_test = XG_test.loc[:, XG_test.columns.isin(['Monocyte_rank'])]
##
dtrain_fold = xgb.DMatrix(X_train, label=y_train)
params = {
    'objective': 'reg:squarederror',  # or another ranking objective
    'random_state' : 42,
    'max_depth' : 6, # default 3
    'learning_rate': 0.1,
    'eval_metric': 'ndcg',  # or another suitable metric for ranking
    # other parameters as needed
}
bst_fold = xgb.train(params, dtrain_fold, num_boost_round=100)
##
dtest_fold = xgb.DMatrix(X_test, label=y_test)
y_pred = bst_fold.predict(dtest_fold)
# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
##
# Calculate spearmans
output_real_predict_df = y_test.copy()
output_real_predict_df["y_pred"] = y_pred
# Add 1-X ranks to the y_pred and rank columns
output_real_predict_df['rank_comparison'] = output_real_predict_df['Monocyte_rank'].rank(ascending=False).astype(int)
output_real_predict_df['y_pred_comparison'] = output_real_predict_df['y_pred'].rank(ascending=False).astype(int)
spearman_correlation = output_real_predict_df[['rank_comparison', 'y_pred_comparison']].corr(method='spearman').iloc[0, 1]

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² score: {r2}")
print(f"Spearman Correlation: {spearman_correlation}")
##
# Saving optimized model dictionary as new variable for testing
#optimized_fc_dict = model_dict

##
# Create starting dictionary for prediction and modify test_input_df to remove imputed
Monocyte_test_input_df = test_input_df.drop(["imputed"],axis=1)
XG_feature_predict_dict = {}
XG_feature_predict_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_predict_dict[model] = []

imputer = SimpleImputer(strategy='median')
test_imp_array = imputer.fit_transform(Monocyte_test_input_df)

for input_ar in test_imp_array:
    XG_feature_predict_dict = forest_quorum_model(input_ar, XG_feature_predict_dict, optimized_fc_dict, 3)

##
XG_feature_predict_df = pd.DataFrame(XG_feature_predict_dict)
XG_feature_predict_subject_id = XG_feature_predict_df["subject_id"].to_numpy()
XG_predict_id_df = XG_feature_predict_df["subject_id"].reset_index() # Gets DF with index and subject_id columns
XG_predict_input_df = XG_feature_predict_df.loc[:, ~XG_feature_predict_df.columns.isin(['subject_id'])] #Keep subject ID in there
d_predict = xgb.DMatrix(XG_predict_input_df)
prediction_final = bst_fold.predict(d_predict)
prediction_dict = {"Monocytes_rank_chage_prediction": prediction_final.tolist()}
prediction_dict["subject_id"] = XG_feature_predict_subject_id.tolist()
prediction_final_df = pd.DataFrame(prediction_dict)
prediction_final_df["Monocyte_rank"] = prediction_final_df["Monocytes_rank_chage_prediction"].rank(ascending=True).astype(int)
prediction_final_df.to_excel("Monocytes_rank_change_prediction.xlsx",index=False)
#
#
# Monocyte Rank prediction finished
#
#
##
#
# Monocytes rank fc training
#
monocyte_training_df = monocyte_training_df.reset_index(drop=True)
# Confirm feature length
rank_columns = [col for col in monocyte_training_df.columns if 'Monocyte_rank' in col]
for subject_id in subject_list[:]:
    optimized_fc_dict[f"model_{subject_id}"] = train_monocytes_models_rank_fold_optimization(subject_id,monocyte_training_df,monocyte_actual_df,0.02)
    clf = optimized_fc_dict[f"model_{subject_id}"][0]
    accuracy = optimized_fc_dict[f"model_{subject_id}"][1]
    X_test_imputed = optimized_fc_dict[f"model_{subject_id}"][3]
    y_test=optimized_fc_dict[f"model_{subject_id}"][5]
    model_rank_value = optimized_fc_dict[f"model_{subject_id}"][6]
    model_rank_fc_value = optimized_fc_dict[f"model_{subject_id}"][7]
    model_df = optimized_fc_dict[f"model_{subject_id}"][9]
    print(f"---- ID {subject_id} ----  Rank fc {model_rank_fc_value} ---- Acc {accuracy}")
    for x in range(len(X_test_imputed)):
        new_x = X_test_imputed[x]
        new_x_rs = new_x.reshape(1,-1) # Reshape like this when it's just a single sample
        new_pred = clf.predict(new_x_rs) # Gives array with prediction
        new_probability = clf.predict_proba(new_x_rs) # Gives array with probability it's in class 0 vs class 1
        true_prediction= new_pred[0] == y_test.reset_index(drop=True)[x]
        true_rank_index = y_test.reset_index()
        true_rank_index = true_rank_index.loc[x,"index"]
        true_subject_id = model_df.loc[true_rank_index,["subject_id",'Monocyte_rank']][0]
        true_rank = model_df.loc[true_rank_index, ["subject_id", 'Monocyte_rank']][1]
        #print(f"Input {x} | Prediction {new_pred} | Probability {new_probability} | {true_prediction} | True rank {true_rank} | Subject id {true_subject_id}")
##
# Create starting dictionary
XG_feature_dict = {}
XG_feature_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_dict[model] = []
imputer = SimpleImputer(strategy='median')
train_imp_array = imputer.fit_transform(monocyte_training_df)
for input_ar in train_imp_array:
    XG_feature_fc_dict = forest_quorum_model(input_ar, XG_feature_dict, optimized_fc_dict, 3)
XG_feature_df = pd.DataFrame(XG_feature_fc_dict)
XG_feature_df_actual = pd.merge(XG_feature_df,monocyte_actual_df,on="subject_id",how="left")
##
# We got it! time for XGboost!
# First subset dataset
XG_rank_fc_df = XG_feature_df_actual.drop(['Monocyte_rank'],axis=1)
gss = GroupShuffleSplit(test_size=.02, n_splits=1, random_state = 7).split(XG_rank_fc_df, groups=XG_rank_fc_df['subject_id'])


X_train_inds, X_test_inds = next(gss)
XG_train = XG_rank_fc_df.iloc[X_train_inds]
X_train = XG_train.loc[:, ~XG_train.columns.isin(['subject_id','Monocytes_rank_fc'])]
y_train = XG_train.loc[:, XG_train.columns.isin(['Monocytes_rank_fc'])]

XG_test = XG_rank_fc_df.iloc[X_test_inds]
X_test = XG_test.loc[:, ~XG_test.columns.isin(['subject_id','Monocytes_rank_fc'])]
y_test = XG_test.loc[:, XG_test.columns.isin(['Monocytes_rank_fc'])]
##

dtrain_fold = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'reg:squarederror',  # or another ranking objective
    'random_state' : 42,
    'max_depth' : 6, # default 3
    'learning_rate': 0.1,
    'eval_metric': 'ndcg',  # or another suitable metric for ranking
    # other parameters as needed
}

bst_fold = xgb.train(params, dtrain_fold, num_boost_round=100)
##

dtest_fold = xgb.DMatrix(X_test, label=y_test)
y_pred = bst_fold.predict(dtest_fold)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
##
# Calculate spearmans
output_real_predict_df = y_test.copy()
output_real_predict_df["y_pred"] = y_pred
# Add 1-X ranks to the y_pred and rank columns
output_real_predict_df['rank_comparison'] = output_real_predict_df['Monocytes_rank_fc'].rank(ascending=False).astype(int)
output_real_predict_df['y_pred_comparison'] = output_real_predict_df['y_pred'].rank(ascending=False).astype(int)
spearman_correlation = output_real_predict_df[['rank_comparison', 'y_pred_comparison']].corr(method='spearman').iloc[0, 1]

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² score: {r2}")
print(f"Spearman Correlation: {spearman_correlation}")
##
# Saving optimized model dictionary as new variable for testing
#optimized_fc_dict = model_dict

##

# Create starting dictionary for prediction and modify test_input_df to remove imputed
Monocyte_test_input_df = test_input_df.drop(["imputed"],axis=1)
XG_feature_predict_dict = {}
XG_feature_predict_dict["subject_id"] = []
for model in optimized_fc_dict.keys():
    XG_feature_predict_dict[model] = []

imputer = SimpleImputer(strategy='median')
test_imp_array = imputer.fit_transform(Monocyte_test_input_df)

for input_ar in test_imp_array:
    XG_feature_predict_dict = forest_quorum_model(input_ar, XG_feature_predict_dict, optimized_fc_dict, 3)

##
XG_feature_predict_df = pd.DataFrame(XG_feature_predict_dict)
XG_feature_predict_subject_id = XG_feature_predict_df["subject_id"].to_numpy()
XG_predict_id_df = XG_feature_predict_df["subject_id"].reset_index() # Gets DF with index and subject_id columns
XG_predict_input_df = XG_feature_predict_df.loc[:, ~XG_feature_predict_df.columns.isin(['subject_id'])] #Keep subject ID in there
d_predict = xgb.DMatrix(XG_predict_input_df)
prediction_final = bst_fold.predict(d_predict)


prediction_dict = {"Monocytes_rank_fold_chage_prediction": prediction_final.tolist()}
prediction_dict["subject_id"] = XG_feature_predict_subject_id.tolist()
prediction_final_df = pd.DataFrame(prediction_dict)
prediction_final_df["Monocytes_rank_fc"] = prediction_final_df["Monocytes_rank_fold_chage_prediction"].rank(ascending=True).astype(int)
prediction_final_df.to_excel("Monocytes_rank_fold_change_prediction.xlsx",index=False)