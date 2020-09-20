
import pickle

import constant, conf
import utils

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

    sample_skeleton_f.close()
    sampled_result_f_o.close()

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

def common_features_sample_and_save_v1(
        train_sampled_user_set,
        common_features_path, sample_rate=0.001, sample_result_save_path='./common_features_sampled.csv'):
    """
    测试集采样， 取train_sampled_user_set中的user对应行为记录
    :param train_sampled_user_set:
    :param common_features_path:
    :param sample_rate:
    :param sample_result_save_path:
    :return:
    """
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
                # if user_id % constant.FLOW_NUM < constant.FLOW_NUM * sample_rate:
                if user_id in train_sampled_user_set:
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
    """
    随机采样
    :param common_features_path:
    :param sample_skeleton_path:
    :param sample_rate:
    :param sampled_common_features_path:
    :param sampled_skeleton_path:
    :return:
    """
    sampled_common_features_path = sampled_common_features_path.format(str(sample_rate).split('.')[1])
    sampled_skeleton_path = sampled_skeleton_path.format(str(sample_rate).split('.')[1])
    md5_set = common_features_sample_and_save(common_features_path, sample_rate, sampled_common_features_path)
    skeleton_sample_and_save(sample_skeleton_path, md5_set, sampled_skeleton_path)

def data_sample_v1(
        train_sampled_user_set,
        common_features_path, sample_skeleton_path, sample_rate, sampled_common_features_path, sampled_skeleton_path
):
    """
    用于测试集采样， 采样训练集采样结果中的user的历史行为
    :param train_sampled_user_set:
    :param common_features_path:
    :param sample_skeleton_path:
    :param sample_rate:
    :param sampled_common_features_path:
    :param sampled_skeleton_path:
    :return:
    """
    sampled_common_features_path = sampled_common_features_path.format(str(sample_rate).split('.')[1])
    sampled_skeleton_path = sampled_skeleton_path.format(str(sample_rate).split('.')[1])
    md5_set = common_features_sample_and_save_v1(
        train_sampled_user_set, common_features_path, sample_rate, sampled_common_features_path)
    skeleton_sample_and_save(sample_skeleton_path, md5_set, sampled_skeleton_path)


if __name__ == '__main__':
    # data_sample(
    #     utils.path_combine(conf.train_root_path, conf.raw_train_common_features_file_name),
    #     utils.path_combine(conf.train_root_path, conf.raw_train_sample_skeleton_file_name),
    #     conf.sampling_rate,
    #     utils.path_combine(conf.train_root_path, conf.raw_train_sampled_common_features_file_name),
    #     utils.path_combine(conf.train_root_path, conf.raw_train_sampled_skeleton_file_name)
    # )
    pass