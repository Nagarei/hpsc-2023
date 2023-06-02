import numpy as np
import matplotlib.pyplot as plt

nx = int(input())
ny = int(input())
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

nt = int(input())
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
def input_arr(arr: np.ndarray):
  for j in range(ny):
    for i in range(nx):
      arr[j, i] = float(input())
      
for t in range(nt):
  input_arr(u)
  input_arr(v)
  input_arr(p)
  plt.clf()
  plt.title(t)
  plt.contourf(X, Y, p, alpha=0.5, cmap=plt.cm.coolwarm)
  plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
  plt.pause(.001)

plt.show()

