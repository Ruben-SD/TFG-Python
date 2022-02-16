import os
folders = [folder[0] for folder in os.walk(".\data")]
[print("["+str(i)+"]", folder) for i, folder in enumerate(folders)]
path = folders[int(input("Enter file index:"))]


import numpy as np
posData = np.loadtxt(path + '/posData.csv', delimiter=',')
realPosData = np.loadtxt(path + '/realPosData.csv', delimiter=',')
timeData = np.loadtxt(path + '/timeData.csv', delimiter=',')


import matplotlib.pyplot as plt
plt.plot(timeData, posData, 'b')
plt.fill_between(timeData, realPosData - 0.5, realPosData + 0.5, facecolor='black')
plt.xlabel("Time (s)")
plt.ylabel("Position (cm)")
plt.title("Position over time")
plt.yticks(np.arange(0, 64, 2))
plt.grid()
plt.show()

print(timeData[-1])
avgError = np.sum(np.abs(realPosData - posData))/len(timeData)
print("Error = " + str(avgError))
