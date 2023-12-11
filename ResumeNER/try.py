with open('train4.char.bmes', 'r', encoding='utf-8') as f:
    data = f.readlines()
    f.close()

with open('dev4.char.bmes', 'w', encoding='utf-8') as dev:
    with open('train4.char.bmes', 'w', encoding='utf-8') as train:
        for i in range(len(data)):
            if i < (len(data) / 10):
                dev.write(data[i])
            else:
                train.write(data[i])
# 111