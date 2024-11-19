import os
def process_paths(paths, data_mode, n_obj):
    processed_paths = []
    if data_mode == "gt":
        # 给每个路径前面加上 "gt"
        processed_paths = ["gt_3_" + path for path in paths[:n_obj]]
    elif data_mode == "mv":
        # 对于每个路径，复制一份分别加上 "gt" 和 "mv"
        for path in paths[:n_obj]:
            processed_paths.append("gt_3_" + path)
            processed_paths.append("mv_3_" + path)

    # for 2d data
    processed_paths = processed_paths + ["mv_2_" + path for path in paths[n_obj:]]
    
    return processed_paths


def get_pick_index(input_image_path):
    for filename in os.listdir(input_image_path):
        if filename.startswith('032'):
            pick_index = int(filename.split('_')[-1].split('.')[0])
            return pick_index
    return 0
