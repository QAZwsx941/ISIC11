def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    # """
    # 在循环中调用以创建终端进度条
    # @params:
    #     iteration   - 必需  : 当前迭代次数 (整数)
    #     total       - 必需  : 总迭代次数 (整数)
    #     prefix      - 可选  : 前缀字符串 (字符串)
    #     suffix      - 可选  : 后缀字符串 (字符串)
    #     decimals    - 可选  : 百分比完整度中保留的小数位数 (正整数)
    #     length      - 可选  : 进度条的字符长度 (整数)
    #     fill        - 可选  : 进度条填充字符 (字符串)
    # """
    # percent
    # 表示进度条的百分比，通过将当前迭代次数除以总迭代次数得到，并按照指定的小数位数格式化为字符串。
    #     # filledLength
    #     # 表示填充长度，即根据当前迭代次数和总迭代次数，计算出应该填充的字符数。
    #     # bar
    # 表示进度条字符串，通过将填充字符乘以填充长度得到填充部分，再用连字符 - 填充剩余部分，以形成完整的进度条字符串。
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # 当完成时打印换行符
    if iteration == total:
        print()
