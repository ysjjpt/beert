#inference.py
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# 加载保存的模型和分词器
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# 推理函数
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# 测试推理
text = "这个酒店价格很贵，我不喜欢。"
predicted_class = predict(text)
print(f"Predicted class: {predicted_class}")
