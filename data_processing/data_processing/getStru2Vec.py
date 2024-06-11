import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

# 定义一个函数用于批量解析Python查询
def multipro_python_query(data_list):
    # 对传入的数据列表中的每一行进行Python查询解析，并返回解析结果列表
    return [python_query_parse(line) for line in data_list]

# 定义一个函数用于批量解析Python代码
def multipro_python_code(data_list):
    # 对传入的数据列表中的每一行进行Python代码解析，并返回解析结果列表
    return [python_code_parse(line) for line in data_list]

# 定义一个函数用于批量处理Python上下文
def multipro_python_context(data_list):
    # 初始化一个空列表用于存放处理结果
    result = []
    # 遍历数据列表中的每一行
    for line in data_list:
        # 如果行内容为'-10000'，表示特定的空上下文标志
        if line == '-10000':
            # 将['-10000']添加到结果列表中
            result.append(['-10000'])
        else:
            # 否则，调用python_context_parse函数解析上下文，并将结果添加到列表中
            result.append(python_context_parse(line))
    # 返回处理后的上下文数据列表
    return result
# 定义一个函数用于批量解析SQL语言的查询
def multipro_sqlang_query(data_list):
    # 对传入的数据列表中的每一项使用sqlang_query_parse函数进行处理，处理后的结果作为列表返回
    return [sqlang_query_parse(line) for line in data_list]

# 定义一个函数用于批量解析SQL语言的代码
def multipro_sqlang_code(data_list):
    # 对传入的数据列表中的每一项使用sqlang_code_parse函数进行处理，处理后的结果作为列表返回
    return [sqlang_code_parse(line) for line in data_list]

# 定义一个函数用于批量处理SQL语言的上下文
def multipro_sqlang_context(data_list):
    # 创建一个空列表，用于存储处理后的结果
    result = []
    # 遍历输入的数据列表
    for line in data_list:
        # 如果列表中的数据是'-10000'，则直接添加到结果列表中（代表了特定的场景或标记）
        if line == '-10000':
            result.append(['-10000'])
        else:
            # 否则，使用sqlang_context_parse函数处理数据，并添加到结果列表中
            result.append(sqlang_context_parse(line))
    # 返回处理后的上下文数据列表
    return result
def parse(data_list, split_num, context_func, query_func, code_func):
    pool = multiprocessing.Pool()
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()

    return context_data, query_data, code_data

def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]

    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
