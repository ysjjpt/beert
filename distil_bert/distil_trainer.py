from easy_bert.bert4classification.classification_trainer import ClassificationTrainer
from easy_bert.tinybert_distiller import TinyBertDistiller


#texts = ['这部电影太棒了！', '非常失望，浪费时间', '演员表演得很好。', '电影很好看', '导演不行啊，编剧也没有逻辑']
texts = ['这电影棒！', '浪费时间', '表演得很好。', '电影很好看', '导演不行啊，编剧也没有逻辑']
labels = ['正面', '负面', '正面', '正面', '负面']

teacher_pretrained, teacher_model_dir = './models/chinese-roberta-wwm-ext', './test_distil/test_models'
student_pretrained, student_model_dir = './models/TinyBERT_4L_zh', './test_distil/test_models2'

# 训练老师模型
trainer = ClassificationTrainer(teacher_pretrained, teacher_model_dir)
trainer.train(texts, labels, validate_texts=texts, validate_labels=labels, batch_size=2, epoch=20)

# 蒸馏学生
distiller = TinyBertDistiller(
   teacher_pretrained, teacher_model_dir, student_pretrained, student_model_dir,
   task='classification'
)
distiller.distill_train(texts, labels, max_len=20, epoch=30, batch_size=8)


new_texts = [
        "这部电影真的很不错，值得一看！",
        "太无聊了，不推荐。",
        "演员的表演非常出色，情节也很吸引人。"
    ]
from easy_bert.bert4classification.classification_predictor import ClassificationPredictor
predictor = ClassificationPredictor(student_pretrained, student_model_dir)
with open('predictor.txt', 'w', encoding='utf-8') as f:
    predictions = predictor.predict(texts)
    for text, prediction in zip(texts, predictions):
        f.write(f"评论: {text}\n")
        f.write(f"预测情感: {prediction}\n")
        f.write("-" * 50 + "\n")
#predictor = ClassificationPredictor(teacher_pretrained, teacher_model_dir)
#print(predictor.predict(texts))
