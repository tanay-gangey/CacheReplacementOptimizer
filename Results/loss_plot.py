import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("LSTMMultiCells.txt", sep=' ',header = None)
print(df.head())


losses = df[4]
steps = len(losses)
print(losses)

plt.plot(losses)
plt.show()