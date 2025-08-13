import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# 教师模型（较大的Transformer模型）
teacher_model_name = 'hfl/chinese-roberta-wwm-ext'
teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_name)
teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_name, num_labels=2)

# 学生模型（较小的Transformer模型）
student_model_name = 'huawei-noah/TinyBERT_General_4L_312D'
student_tokenizer = BertTokenizer.from_pretrained(student_model_name)
student_model = BertForSequenceClassification.from_pretrained(student_model_name, num_labels=2)


texts = ['这部电影太棒了！', '非常失望，浪费时间', '演员表演得很好。', '电影很好看', '导演不行啊，编剧也没有逻辑']
labels = [1, 0, 1, 1, 0]  # 假设1表示正面，0表示负面

# 创建数据集
dataset = TextClassificationDataset(texts, labels, student_tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


def distillation_loss(teacher_logits, student_logits, labels, alpha=0.5, temperature=2.0):
    # 软目标损失
    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                         F.softmax(teacher_logits / temperature, dim=1),
                         reduction='batchmean')
    
    # 硬目标损失
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 总损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss

# 将教师模型设置为评估模式
teacher_model.eval()

# 将学生模型设置为训练模式
student_model.train()

# 定义优化器
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# 训练学生模型
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # 获取教师模型的输出
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask).logits

        # 获取学生模型的输出
        student_outputs = student_model(input_ids, attention_mask=attention_mask).logits

        # 计算损失
        loss = distillation_loss(teacher_outputs, student_outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# 保存训练好的学生模型
student_model.save_pretrained('./saved_distilled_model')
student_tokenizer.save_pretrained('./saved_distilled_model')

# 定义推理函数
def predict_sentiment(text, model, tokenizer):
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    inputs = tokenizer(text, 
                      truncation=True, 
                      padding='max_length', 
                      max_length=128, 
                      return_tensors='pt')
    
    # 进行推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

# 测试推理
test_texts = [
    "这部电影太棒了，演员表演非常出色！",
    "服务态度很差，完全不推荐。",
    "产品质量一般，价格偏贵。",
    "这个餐厅的菜品非常美味，环境也很好。"
]

print("\n开始进行情感分析推理：")
print("-" * 50)

for text in test_texts:
    predicted_class, confidence = predict_sentiment(text, student_model, student_tokenizer)
    sentiment = "正面" if predicted_class == 1 else "负面"
    print(f"文本: {text}")
    print(f"情感: {sentiment}")
    print(f"置信度: {confidence:.4f}")
    print("-" * 50)