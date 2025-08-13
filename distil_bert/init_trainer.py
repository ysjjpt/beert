pretrained_model_dir, your_model_dir = './models/chinese-roberta-wwm-ext', './test_models'
def init_trainer():
    """初始化 Easy-BERT 分类器"""
    from easy_bert.bert4classification.classification_trainer import ClassificationTrainer

    # 定义标签
    labels = ['正面', '负面']

    # 初始化分类训练器
    trainer = ClassificationTrainer(pretrained_model_dir, your_model_dir)
    return trainer

def train_model(trainer):
    texts = ['这部电影太棒了！', '非常失望，浪费时间', '演员表演得很好。', '电影很好看', '导演不行啊，编剧也没有逻辑']
    labels = ['正面', '负面', '正面', '正面', '负面']
    # 训练模型
    trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=5)

if __name__ == '__main__':
    trainer = init_trainer()
    train_model(trainer)
