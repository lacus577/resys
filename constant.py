# field id
FI_USER_ID = '101'

# 样本采样，均匀分成100份
FLOW_NUM = 100


# 表头
CLICK = 'click'
USER_ID = 'UserID'
ITEM_ID = 'ItemID'
ITEM_COLUMNS = ['SampleID', 'ItemID', 'ItemCateID', 'ItemShopID', 'ItemIntentID', 'ItemBrandID']
USER_COLUMS = ['SampleID', 'UserID', 'UserCateActions', 'UserShopActions', 'UserBrandActions', 'UserIntentActions', 'UserCate1', 'UserCate2', 'UserGender', 'UserAge', 'UserComsumptionLevel1', 'UserComsumptionLevel2', 'UserWorking', 'UserGeo']
ACTION_COLUMNS = ['SampleID', 'UserID', 'ItemID', 'ComposeCateAndActions', 'ComposeShopAndActions', 'ComposeBrandAndActions', 'ComposeIntentAndActions', 'SceneID', 'click', 'buy']

# 原始数据id和名字对应
field_id2name = {'205':'ItemID',
                 '206':'ItemCateID',
                 '207':'ItemShopID',
                 '210':'ItemIntentID',
                 '216':'ItemBrandID',
                 '508':'ComposeCateAndActions',
                 '509':'ComposeShopAndActions',
                 '702':'ComposeBrandAndActions',
                 '853':'ComposeIntentAndActions',
                 '301':'SceneID',
                 '101':'UserID',
                 '109_14':'UserCateActions',
                 '110_14':'UserShopActions',
                 '127_14':'UserBrandActions',
                 '150_14':'UserIntentActions',
                 '121':'UserCate1',
                 '122':'UserCate2',
                 '124':'UserGender',
                 '125':'UserAge',
                 '126':'UserComsumptionLevel1',
                 '127':'UserComsumptionLevel2',
                 '128':'UserWorking',
                 '129':'UserGeo'}

skeletion_field_id2name = {'205':'ItemID',
                 '206':'ItemCateID',
                 '207':'ItemShopID',
                 '210':'ItemIntentID',
                 '216':'ItemBrandID',
                 '508':'ComposeCateAndActions',
                 '509':'ComposeShopAndActions',
                 '702':'ComposeBrandAndActions',
                 '853':'ComposeIntentAndActions',
                 '301':'SceneID'}

common_features_field_id2name = {'101':'UserID',
                 '109_14':'UserCateActions',
                 '110_14':'UserShopActions',
                 '127_14':'UserBrandActions',
                 '150_14':'UserIntentActions',
                 '121':'UserCate1',
                 '122':'UserCate2',
                 '124':'UserGender',
                 '125':'UserAge',
                 '126':'UserComsumptionLevel1',
                 '127':'UserComsumptionLevel2',
                 '128':'UserWorking',
                 '129':'UserGeo'}