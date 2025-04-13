from project.backend.models.preprocess import preprocess_data
from project.backend.models.model import model

print(model.predict(preprocess_data(['CCO'])))
# print(preprocess_data(['C1(NON=C1C2=CC=CC=C2)=N']).shape)


