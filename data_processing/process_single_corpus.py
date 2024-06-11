import pickle
from collections import Counter
##第一步
#加载pickle文件的数据
def load_pickle(filename):
    # 打开指定的pickle文件，使用二进制读取模式 ('rb')
    with open(filename, 'rb') as f:
        # 从文件中加载数据，并使用 'iso-8859-1' 编码解码数据
        data = pickle.load(f, encoding='iso-8859-1')
    # 返回加载的数据


def split_data(total_data, qids):
    """
    将数据分为两个部分：仅出现一次的数据和多次出现的数据。

    Args:
        total_data (list): 包含所有数据的列表。
        qids (list): 包含每个数据唯一标识符的列表。

    Returns:
        tuple: 包含两个列表，一个是仅出现一次的数据，另一个是多次出现的数据。
    """
    # 使用 Counter 统计 qids 中每个唯一标识符的出现次数
    result = Counter(qids)

    # 初始化存储单次和多次出现数据的列表
    total_data_single = []
    total_data_multiple = []

    # 遍历所有数据，将数据根据出现次数分别存储在不同的列表中
    for data in total_data:
        if result[data[0][0]] == 1:  # 如果该标识符仅出现一次
            total_data_single.append(data)
        else:  # 如果该标识符出现多次
            total_data_multiple.append(data)

    return total_data_single, total_data_multiple


def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
    处理 STaqC 数据集，将数据根据唯一标识符的出现次数进行分割并保存。

    Args:
        filepath (str): 输入的数据文件路径，包含所有数据。
        save_single_path (str): 保存仅出现一次的数据的文件路径。
        save_multiple_path (str): 保存多次出现的数据的文件路径。
    """
    # 读取输入文件中的数据
    with open(filepath, 'r') as f:
        total_data = eval(f.read())

    # 提取每条数据的唯一标识符
    qids = [data[0][0] for data in total_data]

    # 通过 split_data 函数分割数据
    total_data_single, total_data_multiple = split_data(total_data, qids)

    # 将分割后的数据保存到指定文件
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
    处理大型数据集，将数据根据唯一标识符的出现次数进行分割并保存。

    Args:
        filepath (str): 输入的数据文件路径（pickle格式）。
        save_single_path (str): 保存仅出现一次的数据的文件路径（pickle格式）。
        save_multiple_path (str): 保存多次出现的数据的文件路径（pickle格式）。
    """
    total_data = load_pickle(filepath)  # 从pickle文件加载数据
    qids = [data[0][0] for data in total_data]  # 提取每条数据的唯一标识符
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 分割数据

    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)  # 保存仅出现一次的数据
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)  # 保存多次出现的数据


def single_unlabeled_to_labeled(input_path, output_path):
    """
    将未标记的单次数据转换为带标签的数据，并按一定规则排序。

    Args:
        input_path (str): 输入的数据文件路径（pickle格式）。
        output_path (str): 输出的数据文件路径（文本格式）。
    """
    total_data = load_pickle(input_path)  # 从pickle文件加载数据
    labels = [[data[0], 1] for data in total_data]  # 为每条数据添加标签
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 按照标签排序

    with open(output_path, "w") as f:
        f.write(str(total_data_sort))  # 保存排序后的数据


# 主程序
if __name__ == "__main__":
    # 处理 Python STaqC 数据集
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 处理 SQL STaqC 数据集
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 处理大型 Python 数据集
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 处理大型 SQL 数据集
    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    # 将大型数据集的单次数据转换为带标签并排序后保存
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)

