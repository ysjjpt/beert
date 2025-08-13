#data_processing.py
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

# 加载数据集并划分训练集和验证集
def load_and_process_data(csv_file):
    df = pd.read_csv(csv_file)
    # 确保列名与你的CSV文件中的列名相匹配
    df.dropna(subset=['review'], inplace=True)  # 删除含有 NaN 的行
    df['review'] = df['review'].astype(str)  # 确保所有文本数据都是字符串类型

    train_texts, val_texts, train_labels, val_labels = train_test_split(df['review'], df['label'], test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained('/root/ai_course/bert/bert-models/bert-base-chinese')


    # 分词函数
    def tokenize_data(texts):
        # 确保文本列表中的所有条目都是字符串
        texts = texts.tolist()
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)

    # TODO 查看上面提供的代码 对数据进行处理 给出方法的 return内容
    # TODO 提示： 1. 根据train_model.py引用可知 此方法需要返回4个数据
    #            2. 需要处理的数据是上述代码中未被应用的部分
    #            3. 处理内容是对训练集和验证集进行 分词
    train_encodings = tokenize_data(train_texts)
    val_encodings = tokenize_data(val_texts)
    return train_encodings,val_encodings,train_labels.tolist(), val_labels.tolist()

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
