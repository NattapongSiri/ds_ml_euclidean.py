import json
import numpy as np
import keras
from keras.layers import Dense, Input, LSTM
from keras.models import Model
import sys
import matplotlib.pyplot as plt

with open("processed.txt", "r") as fp:
    all_rec = []
    l_bound = sys.maxsize
    precision = 3
    u_bound = 0
    for line in fp:
        rec = json.loads(line)
        conv = []
        for s in rec: # each sensor in a record
            # 3 digit precision need about 48GB memory to load 5 sensor data
            # v = int(s * 1000)
            # 2 digit precision need 4.8GB memory
            v = int(s * 10 ** precision)
            l_bound = min(v, l_bound)
            u_bound = max(v, u_bound)
            conv.append(v)
        all_rec.append(conv) # add each converted record
    # load all data with line below. caution, it need about 48GB memory
    # shape = [len(all_rec), len(all_rec[0]), u_bound + 1] # There's no extra state so max = upper bound + 1
    
    # load first sensor data only. it took about 10GB memory
    # sample_size = 41
    # step = 41
    look_back = 3
    step = 1
    number_sample = int((len(all_rec) - look_back - 1) / step)
    print((len(all_rec) - look_back - 1) / step)
    input_shape = [number_sample, look_back]
    all_vec = np.zeros(input_shape) 
    labels = np.zeros((number_sample))
    for i in range(0, number_sample): # from 0 to n - sample_size - 1, create sample and label
        # print(i * step + input_size, ":", len(all_rec))
        all_vec[i] = [all_rec[j][0] for j in range(i * step, i * step + look_back)] # assign sensor 0 sampled value to nd_array
        labels[i] = all_rec[(i * step) + look_back][0]
        # the for loop below use for load all sensor data
        # for j in range(0, len(all_rec[i])): # each sensor
            # all_vec[i][j][all_rec[i][j]] = 1
    
    print(all_vec)
    print(labels)
    # convert all_vec to actual vec
    all_vec -= l_bound
    all_vec /= u_bound
    x = all_vec.reshape(number_sample, look_back, 1)

    # convert labels to one hot
    # y = keras.utils.to_categorical(labels)

    # convert all labels to vec, each with value between 0-1
    y = labels - l_bound
    y /= u_bound

    # sensors_inputs = [Input(shape=(len(all_vec, u_bound + 1))) for i in range(0, len(all_vec[0])]
    sensor_input = Input(shape=(look_back, 1), dtype=np.float32, name="sensor1") # test case with single sensor
    lstm_state = LSTM(6, activation="sigmoid")(sensor_input)
    output = Dense(1, name="output")(lstm_state)

    model = Model(inputs=[sensor_input], outputs=[output])
    # adam_op = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    epoch = 40
    lr = 0.01
    momentum = 0.8
    decay = lr / epoch
    batch_size = 2
    sgd_op = keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True, clipvalue=1)
    model.compile(optimizer=sgd_op, loss="mean_squared_error", metrics=["mse"])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    model.fit(x, y, epochs=epoch, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop])

    model_cfg = model.to_json()
    with open("model.json", "w") as model_fp:
        model_fp.write(model_cfg)
    model.save_weights("model.h5")
    with open("norm_param.json", "w") as norm_fp:
        norm_fp.write(json.dumps({
            "lowest": l_bound, 
            "highest": u_bound, 
            "precision": precision,
            "look_back" : look_back}))