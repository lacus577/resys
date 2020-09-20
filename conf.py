import os

# valid集合比例
valid_rate = 0.2

# 样本采样比例
sampling_rate = 0.05

# 每路召回策略召回数量
itemcf_matching_num = 1000

# -------------------------- 特征 --------------------------
# 正负样本比例
neg_pos_rate = 20

# gbdt+lr baseline特征：
baseline_features_columns = [
    'UserCate1', 'UserCate2', 'UserGender', 'UserAge', 'UserComsumptionLevel1',
    'UserComsumptionLevel2', 'UserWorking', 'UserGeo',
    'ItemCateID', 'ItemShopID', 'ItemBrandID',
    'ComposeCateAndActions', 'ComposeShopAndActions', 'ComposeBrandAndActions', 'SceneID'
]

# -------------------------- 数据存放路径 --------------------------
# 根目录
train_root_path = r'../../data/tb/sample_train'
test_root_path = r'../../data/tb/sample_test'

# 原始数据存放路径
raw_train_common_features_file_name = 'common_features_train.csv'
raw_train_sample_skeleton_file_name = 'sample_skeleton_train.csv'

raw_test_common_features_file_name = r'common_features_test.csv'
raw_test_sample_skeleton_file_name = r'sample_skeleton_test.csv'

# 采样后中间结果存放路径
raw_train_sampled_common_features_file_name = r'sampled_common_features_train_{}.csv'
raw_train_sampled_skeleton_file_name = r'sampled_skeleton_train_{}.csv'

raw_test_sampled_common_features_file_name = r'sampled_common_features_test_{}.csv'
raw_test_sampled_skeleton_file_name = r'sampled_skeleton_test_{}.csv'

# 采样比例不同，生成对应前缀数据集，前缀是采样比例（比如05_item.csv)
item_file_name = r'{}item.csv'
user_file_name = r'{}user.csv'
action_file_name = r'{}action.csv'


# 中间结果缓存路径
root_caching_path = r'./caching/'
# itemcf 相似度矩阵
itemcf_sim_mat = 'itemcf_sim_mat'

# 模型缓存
gbdt_model_save_file_name = 'gbdt_model'
lr_model_save_file_name = 'lr_model'
