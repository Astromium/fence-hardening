from mlc.datasets.dataset_factory import get_dataset
from neris_attack_tf2 import Neris_attack
from tensorflow.keras.models import load_model
import tensorflow as tf
from neris_model_data_utilities import MDModel, sigmoid
import joblib
import pickle
import numpy as np
import timeit

x_test, y_test = np.load('../data/neris/testing_samples.npy'), np.load('../data/neris/testing_labels.npy')
botnet = np.where(y_test == 1)[0]
x_test_botnet, y_test_botnet = x_test[botnet], y_test[botnet]
print(x_test_botnet.shape, y_test_botnet.shape)

model_path = 'C:\\Users\\ayoub.belouadah\\Desktop\\fence_hardening\\realistic_adversarial_hardening\\botnet\\out\\neris\\adv_retrain_10epochs\\fence_model_adv.h5'
#model_path = 'C:\\Users\\ayoub.belouadah\\Desktop\\adversarial_hardening\\botnet\\resources\\model_botnet.h5'
clf = load_model(model_path)
scaler_path = '../data/neris/scaler.pkl'
#with open(scaler_path, 'rb') as f:
scaler = joblib.load(scaler_path)

mins, maxs = np.load('../data/neris/mins.npy'), np.load('../data/neris/maxs.npy')

BATCH_SIZE = x_test_botnet.shape[0]
print(f'batch size {BATCH_SIZE}')
advs = [] 
preds = []
start = timeit.default_timer()
for i in range(BATCH_SIZE):
    atk = Neris_attack(model_path=model_path, iterations=100, distance=12, scaler=scaler, mins=mins, maxs=maxs)
    #adv = atk.run_attack(sample=x_test_botnet[i].reshape(1, -1), label=y_test_botnet[i])
    advs.append(atk.run_attack(sample=scaler.transform(x_test_botnet[i].reshape(1, -1)), label=y_test_botnet[i])[0])
end = timeit.default_timer()
preds = clf.predict(scaler.inverse_transform(np.array(advs)))
arr = [1 if sigmoid(p) > 0.5 else 0 for p in preds]
print(f'SR {(np.array(arr) != y_test_botnet[:BATCH_SIZE]).astype("int").sum() / BATCH_SIZE}')


'''
for i in range(BATCH_SIZE):
    pred = 1 if sigmoid(clf.predict(X_test_botnet[i].reshape(1, -1))[0][0]) > 0.5 else 0
    if pred != 1:
        continue
    cr += 1
    pred = 1 if sigmoid(clf.predict(advs[i].reshape(1, -1))[0][0]) > 0.5 else 0
    if pred != 1:
        sr += 1

print(f'Correct {cr}')
print(f'SR {sr / cr}')
'''
#print(f'Success Rate {(np.array(preds).reshape(-1,) != y_test_botnet[:BATCH_SIZE]).astype("int").sum() / BATCH_SIZE}')
print(f'Exec time {(end - start) / 60}')

BATCH_SIZE = x_test_botnet.shape[0]
preds = clf.predict(x_test_botnet)
arr = [1 if sigmoid(p) > 0.5 else 0 for p in preds]
print(f'BATCH {BATCH_SIZE}')
#print(f'{(np.array(arr) == y_test[:BATCH_SIZE]).astype("int")}')
print(f'Clean Accuracy {(np.array(arr) == y_test_botnet[:BATCH_SIZE]).astype("int").sum() / BATCH_SIZE}')


