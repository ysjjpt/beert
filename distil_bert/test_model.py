pretrained_model_dir, your_model_dir = './models/chinese-roberta-wwm-ext', './test_models'

def save_model():
    """保存模型"""
    from easy_bert.bert4classification.classification_predictor import ClassificationPredictor

    # 加载保存的模型
    predictor = ClassificationPredictor(pretrained_model_dir, your_model_dir)
    return predictor


def inference(predictor):
    """进行推理"""
    # 新的评论
    new_texts = [
        "这部电影真的很不错，值得一看！",
        "太无聊了，不推荐。",
        "演员的表演非常出色，情节也很吸引人。"
    ]

    # 进行预测
    predictions = predictor.predict(new_texts)

    # 打印预测结果
    with open('predictor.txt', 'w', encoding='utf-8') as f:
        for text, prediction in zip(new_texts, predictions):
            f.write(f"评论: {text}\n")
            f.write(f"预测情感: {prediction}\n")
            f.write("-" * 50 + "\n")

if __name__ == '__main__':
    predictor = save_model()
    inference(predictor)
