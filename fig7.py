import numpy as np
import matplotlib.pyplot as plt

x = [-1, 1, 1, 3, 3, -1, -1, -3, -1]
y = [-1, -1, -3, -1, 1, 1, 3, 1, -1]

v = ['top', 'top', 'top', 'center', 'bottom',
     'bottom', 'bottom', 'center']
h = ['center', 'right', 'center', 'left', 'left',
     'left', 'center', 'right']

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.plot(x,y,'k')
plt.text(x[0], y[0], 'P$_1$', va='top', ha='center', fontsize=16)
plt.text(x[1], y[1], 'P$_2$', va='top', ha='right', fontsize=16)
plt.text(x[3], y[3], 'P$_3$', va='center', ha='left', fontsize=16)
plt.text(x[5], y[5], 'P$_4$', va='bottom', ha='left', fontsize=16)
plt.title('GWW1', fontsize=16)
plt.axis('equal')
plt.axis('off')

x = [1, -1, -1, -3, -3, 1, 1, 3, 1]
y = [1, 1, 3, 3, 1, -3, -1, -1, 1]

plt.subplot(1,2,2)
plt.plot(x,y,'k')
plt.text(x[0], y[0], 'P$_1$', va='bottom', ha='center', fontsize=16)
plt.text(x[1], y[1], 'P$_2$', va='bottom', ha='left', fontsize=16)
plt.text(x[4], y[4], 'P$_3$', va='center', ha='right', fontsize=16)
plt.text(x[6], y[6], 'P$_4$', va='top', ha='left', fontsize=16)
plt.title('GWW2', fontsize=16)
plt.axis('equal')
plt.axis('off')

plt.tight_layout()
plt.savefig('fig7.eps')
plt.show()
