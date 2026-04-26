# 余弦相似度：两个向量的积点 除以 两个向量的模长的乘积
# 示例：a = [1, 2], b = [4, 5]，余弦相似度（sqrt=根号） = (1*4 + 2*5) / (sqrt(1*1 + 2*2) * sqrt(4*4 + 5*5)) = 0.9

import numpy


# 向量点积
def get_dot(vec_a, vec_b):
    """
    计算两个向量的点积：两个向量同维度数字的乘机之和
    :param vec_a: 向量a
    :param vec_b: 向量b
    :return: 点积
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("两个向量维度不一致")

    return sum([a * b for a, b in zip(vec_a, vec_b)])


# 向量模长
def get_norm(vec):
    """
    计算向量的模长：向量所有数字的平方和开根号
    :param vec:
    :return:
    """
    return numpy.sqrt(sum([a * a for a in vec]))


def get_cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度
    :param vec_a: 向量a
    :param vec_b: 向量b
    :return: 余弦相似度
    """
    return get_dot(vec_a, vec_b) / (get_norm(vec_a) * get_norm(vec_b))


print(get_cosine_similarity([1, 2], [4, 5]))
