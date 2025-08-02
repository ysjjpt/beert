#train_model.py
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from data_processing import load_and_process_data, ReviewsDataset

# 加载数据
train_encodings, val_encodings, train_labels, val_labels = load_and_process_data('ChnSentiCorp_htl_all.csv')

# 创建 PyTorch Dataset
train_dataset = ReviewsDataset(train_encodings, train_labels)
# TODO 仿照 训练数据train_dataset 的处理， 补全 推理数据val_dataset的代码
val_dataset =  ...

# 加载本地预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained('/root/ai_course/bert/bert-models/bert-base-chinese', num_labels=2)

# 加载本地预训练的分词器
tokenizer = BertTokenizer.from_pretrained('/root/ai_course/bert/bert-models/bert-base-chinese')


# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 模型保存路径
    eval_strategy="epoch",     # 每个epoch后评估
    per_device_train_batch_size=16,  # 每个设备的批次大小
    per_device_eval_batch_size=16,   # 每个设备的评估批次大小
    num_train_epochs=3,              # 训练轮数
    weight_decay=0.01,               # 权重衰减
    use_cpu=True                     # 强制使用 CPU
)

# 使用 Trainer API 进行训练
# TODO 对于上述处理完成的数据 将下面缺失部分补全
trainer = Trainer(
    model=...,
    args=...,
    train_dataset=...,
    eval_dataset=...
)

# 训练模型
trainer.train()

# 保存模型和分词器
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')  # 保存分词器
