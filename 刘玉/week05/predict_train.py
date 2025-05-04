import fasttext


def train_predict(filename, predict=False):
    """
    # 通过服务器终端进行简单的数据预处理
    # 使标点符号与单词分离并统一使用小写字母
    >> wc -l cooking.stackexchange.txt
    >> head -n 12404 cooking.preprocessed.txt > cooking.train.txt
    >> tail -n 3000 cooking.preprocessed.txt > cooking.valid.txt
    """
    model_path = "dist/Fasttext/fasttext_word_cook_model.bin"
    if predict:
        # 加载模型
        model = fasttext.load_model(model_path)

        # 在测试数据上进行评估 - 元组中的每项分别代表, 验证集样本数量, 精度以及召回率
        result = model.test("data/Fasttext/cooking.valid.txt")
        print(f"测试集上的样本数量: {result[0]}")
        print(f"准确率: {result[1]}，召回率: {result[2]}")

        # 对文本进行分类预测
        predicted_label = model.predict("Why not put knives in the dishwasher?")
        print(f"预测的类别: {predicted_label[0][0]}")
        print(f"预测的置信度: {predicted_label[1][0]}")
    else:
        # 训练模型
        model = fasttext.train_supervised(
            input=filename,
            dim=300,
            epoch=10,
            lr=0.1,
        )
        # 保存模型
        model.save_model(model_path)


# 进行文本分类 - https://zhuanlan.zhihu.com/p/575814154
if __name__ == "__main__":
    if False:
        train_predict("data/Fasttext/cooking.train.txt")
    else:
        train_predict("data/Fasttext/cooking.train.txt", True)
