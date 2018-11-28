import json
import matplotlib.pyplot as plt
distances = []
with open("processed.txt", "r") as input_fp:
    for line in input_fp:
        record = json.loads(line)
        distances.append(record[0])

plt.plot(distances[0:100])
plt.show()