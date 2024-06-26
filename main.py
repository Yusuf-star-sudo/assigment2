from sklearn.metrics import accuracy_score
import models.ai4i as ai4i
import matplotlib.pyplot as plt
import pandas

ai4i.from_csv('datasets/ai4i2020.csv')
ai4i.preprocessing()
ai4i.split()
ai4i.fit()
model = ai4i.model

target = model.predict(ai4i.test[0])
result = pandas.DataFrame(ai4i.test[0])
result['Machine failure'] = target

print(result.head())

plt.scatter(range(len(target)), target)
plt.title("Machine Failure Prediction")
plt.xlabel("Accuracy: "+str(accuracy_score(ai4i.test[1], target)*100)+"%")
plt.show()