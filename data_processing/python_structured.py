# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize

from nltk import wordpunct_tokenize
from io import StringIO
# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnler = WordNetLemmatizer()


from nltk.corpus import wordnet

#############################################################################

# 正则表达式模式，用于匹配变量赋值和循环中的变量
PATTERN_VAR_EQUAL = re.compile(r"(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile(r"for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")


def repair_program_io(code):
    """
    修复代码中的交互式输入输出字符，使之成为标准Python代码。

    Args:
        code (str): 输入的代码字符串。

    Returns:
        tuple: 包含修复后的代码和代码块列表。
    """
    # 正则表达式模式，处理两种不同格式的交互式输入输出
    # 第1类：In [1]:, Out [1]: 和 ....:
    pattern_case1_in = re.compile(r"In ?\[\d+]: ?")
    pattern_case1_out = re.compile(r"Out ?\[\d+]: ?")
    pattern_case1_cont = re.compile(r"( )+\.+: ?")

    # 第2类：>>> 和 ...
    pattern_case2_in = re.compile(r">>> ?")
    pattern_case2_cont = re.compile(r"\.\.\. ?")

    # 汇总所有模式
    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    # 分割代码为多行
    lines = code.split("\n")
    # 初始化每行的标记（是否匹配所定义的正则模式）
    lines_flags = [0 for _ in range(len(lines))]

    code_list = []

    # 匹配模式并标记
    for line_idx, line in enumerate(lines):
        for pattern_idx, pattern in enumerate(patterns):
            if re.match(pattern, line):
                lines_flags[line_idx] = pattern_idx + 1
                break

    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # 如果没有匹配到任何模式，则不需要修复
    if lines_flags.count(0) == len(lines_flags):
        repaired_code = code
        code_list = [code]
        bool_repaired = True
    # 如果匹配到第1类或第2类模式
    elif re.match(re.compile(r"(0*1+3*2*0*)+"), lines_flags_string) or \
            re.match(re.compile(r"(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""

        # 处理开头部分不匹配的行
        if lines_flags[0] == 0:
            flag = 0
            while (flag == 0):
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""

        # 处理匹配的行
        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    # 如果未能按上面的模式修复代码，则简单地删除每个输出后的0标记的行
    if not bool_repaired:
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list


def get_vars(ast_root):
    """
    从AST树中提取所有变量名。

    Args:
        ast_root (AST): AST 树的根节点。

    Returns:
        list: 所有变量名的列表，已排序。
    """
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})


def get_vars_heuristics(code):
    """
    通过启发式方法从代码中提取变量名。

    Args:
        code (str): 输入的代码字符串。

    Returns:
        set: 变量名的集合。
    """
    varnames = set()
    # 去掉空行，并将代码分割为多行
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]

    # 最优解析
    start = 0
    end = len(code_lines) - 1
    bool_success = False

    # 尝试解析代码，直到成功解析或到达代码末尾
    while not bool_success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            bool_success = True

    # 将成功解析的变量加入集合
    varnames = varnames.union(set(get_vars(root)))

    # 处理未解析的剩余行
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            # 匹配变量赋值模式（PATTERN_VAR_EQUAL）
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # 去掉 "="
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

            # 匹配循环中的变量模式（PATTERN_VAR_FOR）
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # 去掉 "for" 和 "in"
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))

    return varnames


def PythonParser(code):
    """
    解析Python代码并进行标记化处理，识别变量名等。

    Args:
        code (str): 输入的代码字符串。

    Returns:
        tuple: (标记化后的代码列表，是否存在变量解析错误的标记，是否存在标记解析错误的标记)
    """
    bool_failed_var = False
    bool_failed_token = False

    try:
        root = ast.parse(code)  # 尝试解析代码为AST
        varnames = set(get_vars(root))
    except:
        repaired_code, _ = repair_program_io(code)  # 尝试修复代码
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except:
            bool_failed_var = True
            varnames = get_vars_heuristics(code)  # 使用启发式方法解析变量名

    tokenized_code = []

    def first_trial(_code):
        """
        尝试第一次解析代码。

        Args:
            _code (str): 输入的代码字符串。

        Returns:
            bool: 是否解析成功。
        """
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)

    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1

        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True
                code_lines = code.split("\n")
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token


#############################################################################
# 缩略词处理函数
def revert_abbrev(line):
    """
    还原缩略词。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 还原了缩略词的字符串。
    """
    pat_is = re.compile(r"(it|he|she|that|this|there|here)(\"s)", re.I)
    pat_s1 = re.compile(r"(?<=[a-zA-Z])\"s")
    pat_s2 = re.compile(r"(?<=s)\"s?")
    pat_not = re.compile(r"(?<=[a-zA-Z])n\"t")
    pat_would = re.compile(r"(?<=[a-zA-Z])\"d")
    pat_will = re.compile(r"(?<=[a-zA-Z])\"ll")
    pat_am = re.compile(r"(?<=[I|i])\"m")
    pat_are = re.compile(r"(?<=[a-zA-Z])\"re")
    pat_ve = re.compile(r"(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


# 获取词性函数
def get_wordpos(tag):
    """
    获取单词的词性。

    Args:
        tag (str): 词性标签。

    Returns:
        str: 对应于WordNet的词性。
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# ---------------------子函数1：句子的去冗--------------------
# 处理自然语言行
def process_nl_line(line):
    """
    对输入的自然语言行进行预处理，包括缩略词还原、空格和换行符处理、命名转换和括号内容去除。

    Args:
        line (str): 输入的自然语言行。

    Returns:
        str: 预处理后的行。
    """
    # 还原缩略词
    line = revert_abbrev(line)

    # 处理制表符和换行符
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')

    # 替换多余的空格
    line = re.sub(' +', ' ', line)
    line = line.strip()

    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里的内容
    space = re.compile(r"\([^(|^)]+\)")
    line = re.sub(space, '', line)

    # 去除开始和末尾空格
    line = line.strip()
    return line


# 处理句子和单词
def process_sent_word(line):
    """
    处理输入的句子，进行分词、替换特殊字符和数字、小写化、词性标注和词干提取。

    Args:
        line (str): 输入的句子。

    Returns:
        list: 处理后的单词列表。
    """
    # 找单词和标点符号
    line = re.findall(r"\w+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)

    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)

    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)

    # 替换数字
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)

    # 替换包含字母和数字的字符串
    other = re.compile(r"(?<![A-Z|a-z_])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)

    # 分词
    cut_words = line.split(' ')

    # 全部小写化
    cut_words = [x.lower() for x in cut_words]

    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)

    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])

        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)

        # 词干提取
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


# 去除非常用符号
def filter_all_invachar(line):
    """
    去除字符串中的非常用符号，包括制表符和换行符。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 去除非常用符号后的字符串。
    """
    # 确保输入是字符串
    assert isinstance(line, str)

    # 去除非常用符号，保留字母、数字、下划线、破折号等
    line = re.sub('[^(0-9|a-zA-Z\\-_\'\")\n]+', ' ', line)

    # 处理一些特定符号
    line = re.sub('-+', '-', line)  # 中横线
    line = re.sub('_+', '_', line)  # 下划线
    line = line.replace('|', ' ').replace('¦', ' ')  # 去除横杠

    return line


# 去除部分非常用符号
def filter_part_invachar(line):
    """
    去除字符串中的部分非常用符号，包括制表符和换行符。

    Args:
        line (str): 输入的字符串。

    Returns:
        str: 去除部分非常用符号后的字符串。
    """
    # 去除非常用符号，保留字母、数字、下划线、破折号等
    line = re.sub('[^(0-9|a-zA-Z\\-_\'\")\n]+', ' ', line)

    # 处理一些特定符号
    line = re.sub('-+', '-', line)  # 中横线
    line = re.sub('_+', '_', line)  # 下划线
    line = line.replace('|', ' ').replace('¦', ' ')  # 去除横杠

    return line


########################主函数：代码的tokens#################################
def python_code_parse(line):
    """
    解析输入的Python代码，并将其标记化为tokens。

    Args:
        line (str): 输入的代码行。

    Returns:
        list: token列表
    """
    # 预处理代码行
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)  # 新增加
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
        # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'


########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

#解析输入的自然语言查询，返回一个解析后的单词列表
def python_query_parse(line):
    line = filter_all_invachar(line)  # 调用辅助函数去除所有非常用符号
    line = process_nl_line(line)       # 调用辅助函数处理自然语言行，包括转换命名和去除括号内容
    word_list = process_sent_word(line)  # 分词处理，返回单词列表
    # 分完词后,再去掉括号
    for i in range(0, len(word_list)):
        if re.findall('[()]', word_list[i]):  # 检查单词是否包含括号
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']  # 去掉空白项
    # 解析可能为空

    return word_list  # 返回处理后的单词列表


#解析输入的上下文语句，执行类似的步骤并返回处理后的单词列表。
def python_context_parse(line):
    line = filter_part_invachar(line)  # 调用辅助函数去除部分非常用符号
    # 在这一步的时候驼峰命名被转换成了下划线
    line = process_nl_line(line)   # 调用辅助函数处理自然语言行，包括转换命名和去除括号内容
    print(line)  # 打印处理后的行，便于调试
    word_list = process_sent_word(line)  # 分词处理，返回单词列表
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']  # 去掉空白项
    # 解析可能为空
    return word_list  # 返回处理后的单词列表


#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    # 解析并打印字符串 "change row_height and column_width in libreoffice calc use python tagint"
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    # 解析并打印字符串 'What is the standard way to add N seconds to datetime.time in Python?'
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    # 解析并打印字符串 "Convert INT to VARCHAR SQL 11?"
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    # 解析并打印字符串 'python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'
    print(python_query_parse('python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    # 解析并打印字符串 'How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'
    print(python_context_parse('How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    # 解析并打印字符串 'how do i display records (containing specific) information in sql() 11?'
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    # 解析并打印字符串 'Convert INT to VARCHAR SQL 11?'
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    # 解析并打印字符串代码 'if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'
    print(python_code_parse('if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    # 解析并打印字符串代码 'root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    # 解析并打印字符串代码 'root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    # 解析并打印字符串代码 'n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    # 解析并打印一段较长的字符串代码
    print(python_code_parse("diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    # 解析并打印一段字典遍历代码 'd = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])'
    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    # 解析并打印一段较长的数据表字符串代码
    print(python_code_parse('  #       page  hour count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
