import json
import matplotlib.pyplot as plt

with open("norm_param.json") as params_fp:
    params = json.load(params_fp)

# compare length
comp_len = 100
# raw data
raw_in = []
# linear activation
lin = []
# sigmoid activation
sigm = []
# tanh activation
tanh = []
with open("processed.txt", "r") as p_fp:
    for i in range(0, comp_len):
        line = p_fp.readline()
        record = json.loads(line)
        raw_in.append(record[0]) # sensor 0

with open("predicted_lin.txt", "r") as lin_fp:
    content = lin_fp.read()
    lin = json.loads(content)[0:comp_len]

with open("predicted_sigm.txt", "r") as sigm_fp:
    content = sigm_fp.read()
    sigm = json.loads(content)[0:comp_len]

with open("predicted_tanh.txt", "r") as tanh_fp:
    content = tanh_fp.read()
    tanh = json.loads(content)[0:comp_len]

plt.plot(raw_in, "r-", label="original")
plt.plot(lin, "b-", label="linear")
plt.plot(sigm, "g-", label="sigmoid")
plt.plot(tanh, "y-", label="tanh")
plt.legend(loc="upper left")
plt.show()