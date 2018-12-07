# -*-coding:utf-8-*-
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.mllib.feature import Word2Vec
import jieba
import re

#delete words
# os.environ['PYSPARK_SUBMIT_ARGS'] = "--master mymaster --total-executor 2 --conf "
def merge(list):
    result = []
    for x in range(len(list)):
        result.extend(list[x])
    list.clear()
    return result


def split(line):
    # string = re.sub('[0-9!"#$%&()?@。?★、()（）「」¥…【】《》？“”‘’！[\\]^_`{|}~]+', "", line[0])
    word_list = jieba.cut(line[0])  # 进行中文分词
    # print(word_list)
    ls = []
    for word in word_list:
        # string = re.sub('[0-9!"#$%&()?@。?★、()（）「」¥…【】《》？“”‘’！[\\]^_`{|}~]+', "", line[0])
        if word.encode('utf-8').isalpha():
            # print(word)
            ls.append(word.lower())
        else:
            for i_word in word:
                ls.append(i_word)
    return ls


def combine(line):  # 去除保存结果中的括号和解=解决中文编码显示的问题
    result = ""
    result += line[0] + "\t" + str(line[1])  # 让数字在前，方便统计
    return result


def main(sc):
    spark = SparkSession.builder.appName("Charembedding").config("spark.some.config.option",
                                                                 "Charembedding").getOrCreate()
    # readjson and preprocess
    file_path = '/user/ichongxiang/data/positions/20180518/dedup_json/part-'
    df = spark.read.json('/user/ichongxiang/data/positions/20180518/dedup_json/part-00000').select("requirement",
                                                                                                   "description")
    text_00000 = df.rdd.map(list)
    text_00000 = text_00000.map(lambda r: [r[0] + r[1]])
    inp_all = text_00000.map(split)
    for i in range(1, 300):
        print('Processing input files:%s/%s' % (i, 300))
        i = "%05d" % i
        df = spark.read.json(file_path + str(i)).select("requirement", "description")
        text = df.rdd.map(list)
        text = text.map(lambda r: [r[0] + r[1]])
        inp = text.map(split)
        inp_all = inp_all.union(inp)

    print('Start Traing Word2vec')
    word2vec = Word2Vec()
    model = word2vec.setVectorSize(100).setMinCount(0).setSeed(100000000000000).fit(inp_all)
    w2v_dict = model.getVectors()
    print('Saving Word2vec model vectors')
    w2v_save = open("char_embedding_w2v.csv", 'w')
    for i, v in w2v_dict.items():
        w2v_save.write(str(i))
        w2v_save.write('\t')
        w2v_save.write(str(v))
        w2v_save.write('\n')
    w2v_save.close()
    print("succeed")


if __name__ == "__main__":
    conf = SparkConf().setAppName("CharEmbedding")
    conf.setMaster("local[5]")
    sc = SparkContext(conf=conf)
    main(sc)
