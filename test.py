import keras
from keras.models import Model, model_from_json
import numpy as np
import json

def denormalize(y, exp, highest, lowest):
    y *= highest
    y += lowest
    y /= 10 ** exp
    return y


with open("model.json", "r") as model_fp:
    model_json = model_fp.read()
with open("norm_param.json", "r") as norm_fp:
    params = json.load(norm_fp)
model = model_from_json(model_json)
model.load_weights("model.h5")

sensor_0 = []
sensor_1 = []
with open("processed.txt") as input_fp:
    for line in input_fp:
        record = json.loads(line)
        sensor_0.append((int(record[0] * 10 ** params["precision"]) - params["lowest"]) / params["highest"])
        sensor_1.append((int(record[1] * 10 ** params["precision"]) - params["lowest"]) / params["highest"])
anchor = 0
# first set of value taken from processed.txt
x = np.array(sensor_0[anchor:anchor + params["look_back"]]).reshape(1, params["look_back"], 1)
prediction_len = 100

with open("predicted.txt", "w") as output_fp:
    output.write("[")
    # copy first raw value into output
    str_x = [str(denormalize(v, params["precision"], params["highest"], params["lowest"])) for v in sensor_0[anchor: anchor + params["look_back"]]]
    output_fp.write(",".join(str_x))

    # predict till reach specified length
    for j in range(0, prediction_len):
        y = model.predict(x)
        output_fp.write("," + str(denormalize(y[0][0], params["precision"], params["highest"], params["lowest"])))
        x = np.roll(x, -1, 1) # rotate row 1 by 1
        x[0][params["look_back"] - 1][0] = y # set last vale to predicted value

    output.write("]")