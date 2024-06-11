# softwareEngineering
20211060206 杨雪霜

## 目录

- [一、项目说明](#一项目框架)
- [二、文件说明](#二文件说明)
  - [2.1 getSru2Vec.py文件](#getStru2Vecpy文件)
  - [2.2 embeddings_process.py文件](#embeddings_processpy文件)
  - [2.3 process_single_corpus.py文件](#process_single_corpuspy文件)
  - [2.4 python_structured.py文件](#python_structuredpy文件)
  - [2.5 sqlang_structured.py文件](#sqlang_structuredpy文件)
  - [2.6 word_dict.py文件](#word_dictpy文件)

## 一、项目框架
```
|── hnn_processing  
│     └── embeddings_process.py  
│     └── getStru2Vec.py
│     └── process_single_corpus.py
│     └── python_structured.py
│     └── sqlang_structured.py
│     └── word_dirt.py
```
此仓库通过增加注释来使代码更加容易理解。

## 二、文件说明

### embeddings_process.py 

#### 1. 概述
处理自然语言处理任务中的词向量和数据预处理的一系列操作。通过将词向量转换成二进制格式来提高加载效率，在通过构建新的词典和词向量矩阵来适应特定的任务需求，通过文本到索引的转换来准备模型的输入数据，最终通过序列化处理使得数据集保持一致且便于模型的训练和评估。

#### 2. 具体功能
- `trans_bin`：将文本格式的词向量文件转换成二进制格式。
- `get_new_dict`：根据给定的词列表生成新的词典和对应的词向量矩阵。其中包括特殊标记（如PAD, SOS, EOS, UNK）的处理和嵌入向量的生成，以及对于未在模型中找到向量的词汇的处理。
- `get_index`：根据给定的文本（代码或文本行），利用之前构建的词典，转换成对应的索引序列
- `Serialization`：处理给定的语料，将它们转换为便于模型训练使用的格式。主要包括文本和代码的分割、限长、填充等操作，以及最终序列的组织。
---
### getStru2Vec.py文件

#### 1. 概述
实现了并行分词的功能，解析和处理大批量数据的程序，特别是对Python和SQL编程语言的查询、代码和上下文数据进行解析。

#### 2. 具体功能
- `multipro_python_query`：用于批量解析Python查询、代码和上下文数据。
- `multipro_python_code`：对用于批量解析Python查询、代码和上下文数据。
- `multipro_python_context`：用于批量解析Python查询、代码和上下文数据。
- `multipro_sqlang_query`：用于批量处理SQL语言的上下文
- `multipro_sqlang_code`：用于批量处理SQL语言的上下文
- `multipro_sqlang_context`：用于批量处理SQL语言的上下文
- `parse_python`：接收一批数据和三个特定于编程语言的解析函数，将数据分割成更小的块，并利用多进程池 (multiprocessing.Pool) 并行处理每一块。处理的结果包括上下文、查询和代码数据。
- `main`：主函数，根据语言类型调用相应的解析函数进行分词处理，并将处理结果保存到文件中。
---
### process_single_corpus.py文件

#### 1. 概述
实现了Python代码和自然语言句子的解析和处理功能。

#### 2. 具体功能
- `load_pickle`：加载pickle文件的数据。
- `split_data`：将数据分为两个部分：仅出现一次的数据和多次出现的数据。
- `data_staqc_processing`：处理 STaqC 数据集，将数据根据唯一标识符的出现次数进行分割并保存。
- `data_large_processing`：针对large数据，将数据根据唯一标识符的出现次数进行分割并保存。
- `single_unlable2lable`：将未标记的单次数据转换为带标签的数据，并按一定规则排序。
---

### python_structured.py文件

#### 1. 概述
用于对数据进行处理和转换，包括加载数据、统计问题的单候选和多候选情况，将数据分为单候选和多候选部分，以及将有标签的数据生成为带有标签的形式。

#### 2. 具体功能
- `repair_program_io`：修复代码中的交互式输入输出字符，使之成为标准Python代码。
- `get_vars`：从AST树中提取所有变量名。
- `get_vars_heuristics`：通过启发式方法从代码中提取变量名。
- `PythonParser`：
  """
    解析Python代码并进行标记化处理，识别变量名等。

    Args:
        code (str): 输入的代码字符串。

    Returns:
        tuple: (标记化后的代码列表，是否存在变量解析错误的标记，是否存在标记解析错误的标记)
    """
- `first_trial`：
 """
        尝试第一次解析代码。

        Args:
            _code (str): 输入的代码字符串。

        Returns:
            bool: 是否解析成功。
        """
- `revert_abbrev`：
  """
    还原缩略词。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 还原了缩略词的字符串。
    """
- `get_wordpos`：
 """
    获取单词的词性。

    Args:
        tag (str): 词性标签。

    Returns:
        str: 对应于WordNet的词性。
    """
- `process_nl_line`：
  """
    对输入的自然语言行进行预处理，包括缩略词还原、空格和换行符处理、命名转换和括号内容去除。

    Args:
        line (str): 输入的自然语言行。

    Returns:
        str: 预处理后的行。
    """
- `process_sent_word`：
  """
    处理输入的句子，进行分词、替换特殊字符和数字、小写化、词性标注和词干提取。

    Args:
        line (str): 输入的句子。

    Returns:
        list: 处理后的单词列表。
    """

- `filter_all_invachar`:
"""
    去除字符串中的非常用符号，包括制表符和换行符。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 去除非常用符号后的字符串。
    """
- `filter_part_invachar`:
 """
    去除字符串中的部分非常用符号，包括制表符和换行符。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 去除部分非常用符号后的字符串。
    """
- `python_code_parse`:
 """
    解析输入的Python代码，并将其标记化为tokens。

    Args:
        line (str): 输入的代码行。

    Returns:
        list: token列表
    """
- `python_query_parse`:解析输入的自然语言查询，返回一个解析后的单词列表
- `python_context_parse`:解析输入的上下文语句，执行类似的步骤并返回处理后的单词列表。
  

---

### sqlang_structured.py文件

#### 1. 概述
完成一个SQL语言解析器的功能，用于对SQL代码进行解析和处理。

#### 2. 具体功能
- `sanitizeSql`： 清理和标准化 SQL 语句。
- `tokenizeRegex`：使用正则表达式扫描器对字符串进行分词。
- `parseStrings`： 解析字符串。
- `renameIdentifiers`：重命名标识符（列和表名）。
- `__hash__`:计算哈希值，用于对象的哈希比较。
- `__init__`:初始化 SqlangParser 对象，设置 SQL 语句并进行解析。
- `getTokens`:获取解析后的标记，将标记平铺。
- `removeWhitespaces`： 移除标记列表中的所有空白标记。
- `identifySubQueries`：识别子查询。
- `identifyLiterals`： 识别各种字面量并重新标记它们。
- `identifyFunctions`：识别 SQL 语句中的函数并标记它们。
- `identifyTables`： 识别 SQL 语句中的表并标记它们。
- `__str__`:将解析后的标记列表转换为字符串。
- `parseSql`：获取解析后的 SQL 标记列表。
- ` revert_abbrev`：缩略词处理
- `get_wordpos`：获取词性
- ` process_nl_liner`：句子预处理
- `process_sent_word`：将一个字符串分解成单词，并进行预处理
- `filter_all_invachar`：用于去除几乎所有非字母数字字符
- `filter_part_invachar`：去除非常用符号；防止解析有误
- `sqlang_code_parse`:处理SQL语料中的代码文本
- `sqlang_query_parse`：处理自然语言查询和上下文
- `sqlang_context_parse`:处理自然语言查询和上下文
  

---

### word_dict.py文件

#### 1. 概述
构建词汇表的过程，采用了Python中常用的pickle模块来加载预先保存的数据。

#### 2. 具体功能
- `get_vocab`：建立整个语料库的词汇表，即所有词的集合
- `load_pickle`：从pickle文件中加载数据并返回。
- `vocab_processing`：用于处理和创建最终的词汇表
