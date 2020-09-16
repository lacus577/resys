
import pickle

import constant, conf

def skeleton_sample_and_save(
        sample_skeleton_path, md5_set, sample_result_save_path='./sample_skeleton_sampled.csv'):
    sample_skeleton_f = open(sample_skeleton_path, 'r')
    sampled_result_f_o = open(sample_result_save_path, 'w')
    index = 0
    for line in sample_skeleton_f:
        tokens = line.strip().split(",")
        if tokens[3] in md5_set:
            sampled_result_f_o.write(line)
        index += 1
        if index % 50000 == 0:
            print("current_index:", index)

def common_features_sample_and_save(
        common_features_path, sample_rate=0.001, sample_result_save_path='./common_features_sampled.csv'):
    common_features_f = open(common_features_path, 'r')
    sampled_result_f_o = open(sample_result_save_path, 'w')
    train_uid_set = set()
    train_md5_set = set()
    index = 0
    for line in common_features_f:
        # 原始数据逗号分隔
        tokens = line.strip().split(",")
        feature_arr = tokens[2].split('\001')
        for fea_kv in feature_arr:
            fea_field_id = fea_kv.split('\002')[0]
            fea_id_val = fea_kv.split('\002')[1]
            fea_id = fea_id_val.split('\003')[0]
            # fea_val = fea_id_val.split('\003')[1]
            if fea_field_id == constant.FI_USER_ID:
                user_id = int(fea_id)
                # 用户uid分成64份
                if user_id % constant.FLOW_NUM < constant.FLOW_NUM * sample_rate:
                    sampled_result_f_o.write(line)

                    train_md5_set.add(tokens[0])
                    train_uid_set.add(user_id)
                    break
        index += 1
        if index % 50000 == 0:
            print("current_index:", index)
    common_features_f.close()
    sampled_result_f_o.close()

    print('user num:', len(train_uid_set))
    return train_md5_set

def data_sample(
        common_features_path, sample_skeleton_path, sample_rate, sampled_common_features_path, sampled_skeleton_path):
    sampled_common_features_path = sampled_common_features_path.format(str(sample_rate).split('.')[1])
    sampled_skeleton_path = sampled_skeleton_path.format(str(sample_rate).split('.')[1])
    md5_set = common_features_sample_and_save(common_features_path, sample_rate, sampled_common_features_path)
    skeleton_sample_and_save(sample_skeleton_path, md5_set, sampled_skeleton_path)


if __name__ == '__main__':
    # common_features_path = r'../../../data/tb/sample_train/common_features_train.csv'
    # sample_skeleton_path = r'../../../data/tb/sample_train/sample_skeleton_train.csv'
    data_sample(
        conf.raw_train_common_features_path, conf.raw_train_sample_skeleton_path,
        conf.sampling_rate,
        conf.raw_train_sampled_common_features_path, conf.raw_train_sampled_skeleton_path
    )