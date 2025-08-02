from data_processing import load_and_process_data

# 加载数据
train_encodings, val_encodings, train_labels, val_labels = load_and_process_data('data_csv.csv')

print(f"train_encodings: {train_encodings}")
print(f"val_encodings: {val_encodings}")
print(f"train_labels: {train_labels}")
print(f"val_labels: {val_labels}")