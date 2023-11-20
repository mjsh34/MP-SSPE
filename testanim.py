import matplotlib.pyplot as plt
import numpy as np
import time


x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
line1, = ax1.plot(x, y, 'b-')
line2, = ax2.plot(x, -y, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):
    #line1.set_ydata(np.sin(0.5 * x + phase))
    line1.set_ydata(1)
    line2.set_ydata(np.sin(-0.5 * x + phase))
    fig.canvas.draw()
    fig.canvas.flush_events()
