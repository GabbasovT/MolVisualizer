from catboost import CatBoostRegressor

# Инициализация модели
model = CatBoostRegressor()

model.load_model('catboost_model.cbm')