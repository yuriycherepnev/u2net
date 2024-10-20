import torch
from model import U2NET # или другую модель

# Загрузка данных
state_dict = torch.load('/home/yuriy/projects/u2net/saved_models/u2net/u2net_1728903520/data.pkl', weights_only=False)

# Создание модели
model = U2NET(3, 1)  # Подберите нужные параметры модели

# Загрузка весов
model.load_state_dict(state_dict)

# Сохранение модели в правильном формате
torch.save(model.state_dict(), '2222.pth')