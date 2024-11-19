import json
import random

# 输入JSON文件路径
input_json_path = r'E:\jingyu\InstantMesh\filtered_obj_name.json'

# 输出JSON文件路径
output_val_json_path = r'E:\jingyu\InstantMesh\val_data.json'
output_train_json_path = r'E:\jingyu\InstantMesh\train_data.json'

# 读取JSON数据
with open(input_json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定义验证集的数据数量
val_count = 1000  # 将此值设置为所需的验证样本数量

# 打乱数据顺序
random.shuffle(data["all_objs"])

# 分割数据
val_data = {
    "good_objs": data["good_objs"],
    "all_objs": data["all_objs"][:val_count]
}
train_data = {
    "good_objs": data["good_objs"],
    "all_objs": data["all_objs"][val_count:]
}

# 写入验证集JSON文件
with open(output_val_json_path, 'w', encoding='utf-8') as val_file:
    json.dump(val_data, val_file, ensure_ascii=False, indent=4)

# 写入训练集JSON文件
with open(output_train_json_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

print("数据分割完成。'val_data.json' 和 'train_data.json' 已创建。")
