import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *
##分词对sqlang 和

#对 data_list 中的每一个查询文本使用 python_query_parse 函数进行解析和分词处理
def multipro_python_query(data_list):
    # 对data_list中的每一行调用python_query_parse函数进行解析和分词，然后返回处理后的列表
    return [python_query_parse(line) for line in data_list]

# 对 data_list 中的每一个代码文本使用 python_code_parse 函数进行解析和分词处理。
def multipro_python_code(data_list):
    # 对data_list中的每一行调用python_code_parse函数进行解析和分词，然后返回处理后的列表
    return [python_code_parse(line) for line in data_list]

# 接受上下文文本列表 data_list。
def multipro_python_context(data_list):
    result = []  # 初始化一个空列表用于存储处理结果
    for line in data_list:
        if line == '-10000':  # 如果当前行是特殊标记 '-10000'
            result.append(['-10000'])  # 直接将 ['-10000'] 添加到结果中
        else:
            # 否则对当前行调用 python_context_parse 函数进行解析和分词，并将结果添加到 result 中
            result.append(python_context_parse(line))
    return result  # 返回处理后的结果列表


# 处理 SQL 语料中的查询文本、代码文本和上下文文本。
def multipro_sqlang_query(data_list):
    # 对 data_list 中的每一行调用 sqlang_query_parse 函数进行解析和分词，然后返回处理后的列表
    return [sqlang_query_parse(line) for line in data_list]


# 处理 SQL 语料中的查询文本、代码文本和上下文文本。
def multipro_sqlang_code(data_list):
    # 对 data_list 中的每一行调用 sqlang_code_parse 函数进行解析和分词，然后返回处理后的列表
    return [sqlang_code_parse(line) for line in data_list]


# 处理 SQL 语料中的查询文本、代码文本和上下文文本。
def multipro_sqlang_context(data_list):
    result = []  # 初始化一个空列表用于存储处理结果
    for line in data_list:
        if line == '-10000':  # 如果当前行是特殊标记 '-10000'
            result.append(['-10000'])  # 直接将 ['-10000'] 添加到结果中
        else:
            # 否则对当前行调用 sqlang_context_parse 函数进行解析和分词，并将结果添加到 result 中
            result.append(sqlang_context_parse(line))
    return result  # 返回处理后的结果列表


# 接受数据列表、分割大小、上下文解析函数、查询解析函数和代码解析函数作为参数。
def parse(data_list, split_num, context_func, query_func, code_func):
    # 创建一个进程池
    pool = multiprocessing.Pool()

    # 将数据按 split_num 大小分割成多个子列表
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]

    # 使用进程池调用 context_func 对每个子列表进行处理
    results = pool.map(context_func, split_list)
    # 将子列表中的结果展开成一个大列表
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    # 使用进程池调用 query_func 对每个子列表进行处理
    results = pool.map(query_func, split_list)
    # 将子列表中的结果展开成一个大列表
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    # 使用进程池调用 code_func 对每个子列表进行处理
    results = pool.map(code_func, split_list)
    # 将子列表中的结果展开成一个大列表
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    # 关闭进程池
    pool.close()
    pool.join()

    return context_data, query_data, code_data


# 接受语言类型、分割大小、源文件路径、保存路径、上下文解析函数、查询解析函数和代码解析函数作为参数。
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    # 读取源文件内容
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 使用 parse 函数对数据进行分词处理
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)

    # 提取每条数据的唯一标识符
    qids = [item[0] for item in corpus_lis]

    # 组合所有处理后的数据
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 将处理后的数据保存到文件中
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)


if __name__ == '__main__':
    # Python 语料的数据路径和保存路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # SQL 语料的数据路径和保存路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # Call the main function for Python and SQL
    main('python_type', split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    main('sqlang_type', split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)

    # Large corpus for Python and SQL
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main('python_type', split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)
    main('sqlang_type', split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)