# text,label
# 这部电影太棒了！,正面
# 非常失望，浪费时间。,负面
# 演员表演得很好。,正面


pretrained_model_dir, your_model_dir = './models/chinese-roberta-wwm-ext', './tests/test_model'

def load_dataset():
    """1.准备数据集"""
    import pandas as pd

    # 加载数据集
    data = pd.read_csv('movie_reviews.csv')

    # 查看数据
    print(data.head())
    return data


def init_easy_bert():
    """2.初始化 Easy-BERT 分类器"""
    from easy_bert.bert4classification.classification_trainer import ClassificationTrainer

    # 定义标签
    labels = ['正面', '负面']

    # 初始化分类训练器
    trainer = ClassificationTrainer(pretrained_model_dir='bert-base-chinese', your_model_dir='./model')
    return trainer


def cut_dataset(data, trainer):
    """3.分割数据集为训练集和验证集"""
    from sklearn.model_selection import train_test_split

    # train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # # 训练模型
    # trainer.train(train_texts, train_labels, validate_texts=val_texts, validate_labels=val_labels, batch_size=8, epoch=5)
    texts = ['天气真好', '今天运气很差']
    labels = ['正面', '负面']

    # trainer = ClassificationTrainer(pretrained_model_dir, your_model_dir)
    trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)


def save_model():
    """4.保存模型"""
    from easy_bert.bert4classification.classification_predictor import ClassificationPredictor

    # 加载保存的模型
    predictor = ClassificationPredictor(pretrained_model_dir='bert-base-chinese', model_dir='./model')
    return predictor


def inference(predictor):
    """5.进行推理"""
    # 新的评论
    new_texts = [
        "这部电影真的很不错，值得一看！",
        "太无聊了，不推荐。",
        "演员的表演非常出色，情节也很吸引人。"
    ]

    # 进行预测
    predictions = predictor.predict(new_texts)

    # 打印预测结果
    for text, prediction in zip(new_texts, predictions):
        print(f"评论: {text}")
        print(f"预测情感: {prediction}")


if __name__ == '__main__':
    data = load_dataset()
    trainer = init_easy_bert()
    cut_dataset(data, trainer)
    predictor = save_model()
    inference(predictor)



from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.tinybert_distiller import TinyBertDistiller

texts = ['天气真好', '今天运气很差']
labels = ['正面', '负面']

teacher_pretrained, teacher_model_dir = './models/chinese-roberta-wwm-ext', './tests/test_model'
student_pretrained, student_model_dir = './models/TinyBERT_4L_zh', './tests/test_model2'

# 训练老师模型
trainer = ClassificationTrainer(teacher_pretrained, teacher_model_dir)
trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

# 蒸馏学生
distiller = TinyBertDistiller(
    teacher_pretrained, teacher_model_dir, student_pretrained, student_model_dir,
    task='classification'
)
distiller.distill_train(texts, labels, max_len=20, epoch=20, batch_size=2)
