# 样本采样比例
sampling_rate = 0.05

# 每路召回策略召回数量
itemcf_matching_num = 1000

# -------------------- 特征 --------------------
# 正负样本比例
neg_pos_rate = 20

# gbdt+lr baseline特征：
baseline_features_columns = [
    'UserCate1', 'UserCate2', 'UserGender', 'UserAge', 'UserComsumptionLevel1',
    'UserComsumptionLevel2', 'UserWorking', 'UserGeo',
    'ItemCateID', 'ItemShopID', 'ItemBrandID',
    'ComposeCateAndActions', 'ComposeShopAndActions', 'ComposeBrandAndActions', 'SceneID'
]

# 中间结果缓存路径
root_caching_path = r'../caching/'
# itemcf 相似度矩阵
itemcf_sim_mat = 'itemcf_sim_mat'

# 模型缓存
gbdt_model_save_path = r'../caching/gbdt_model'
lr_model_save_path = r'../caching/lr_model'

# 数据存放路径
raw_train_common_features_path = r'../../../data/tb/sample_train/common_features_train.csv'
raw_train_sample_skeleton_path = r'../../../data/tb/sample_train/sample_skeleton_train.csv'
raw_train_sampled_common_features_path = r'../../../data/tb/sample_train/sampled_common_features_train_{}.csv'
raw_train_sampled_skeleton_path = r'../../../data/tb/sample_train/sampled_skeleton_train_{}.csv'
train_common_features_path = r'../../../data/tb/sample_train/common_features_train_dataframe.csv'
train_sample_skeleton_path = r'../../../data/tb/sample_train/sample_skeleton_train_dataframe.csv'

train_item_path = r'../../../data/tb/sample_train/item.csv'
train_user_path = r'../../../data/tb/sample_train/user.csv'
train_action_path = r'../../../data/tb/sample_train/action.csv'

raw_test_common_features_path = r'../../../data/tb/sample_test/common_features_test.csv'
raw_test_sample_skeleton_path = r'../../../data/tb/sample_test/sample_skeleton_test.csv'
raw_test_sampled_common_features_path = r'../../../data/tb/sample_test/sampled_common_features_test_{}.csv'
raw_test_sampled_skeleton_path = r'../../../data/tb/sample_test/sampled_skeleton_test_{}.csv'

test_item_path = r'../../../data/tb/sample_test/item.csv'
test_user_path = r'../../../data/tb/sample_test/user.csv'
test_action_path = r'../../../data/tb/sample_test/action.csv'
