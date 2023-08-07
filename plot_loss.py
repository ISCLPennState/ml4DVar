import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

dir = '/eagle/MDClimSim/awikner/climax_4dvar_troy/data/climaX'
cycles = np.arange(17)
end_loss = []
for cycle in cycles:
    files = natsorted([file for file in os.listdir(dir) if 'loss_comps_cycle%d' % cycle in file])
    end_loss.append(np.load(os.path.join(dir, files[-1])))

end_loss = np.array(end_loss)
plt.semilogy(np.sum(end_loss, axis = 1), label = 'Total')
plt.semilogy(np.sum(end_loss[:, :27], axis = 1), label = 'Background')
plt.semilogy(np.sum(end_loss[:,27:27*2], axis = 1), label = 'Background HF')
plt.semilogy(end_loss[:, 27*2 + 8], label = 'Obs. U Wind 250')
plt.grid()
plt.legend()
plt.show()

"""
loss_comps = []
for file in files:
    loss_comps.append(np.load(os.path.join(dir, file)))

loss_comps = np.array(loss_comps)

plt.semilogy(np.sum(loss_comps, axis = 1), label = 'Total')
plt.semilogy(np.sum(loss_comps[:, :27], axis = 1), label = 'Background')
plt.semilogy(np.sum(loss_comps[:,27:27*2], axis = 1), label = 'Background HF')
plt.semilogy(loss_comps[:, 27*2 + 8], label = 'Obs. U Wind 250')
plt.ylim(1e1, 1e7)
plt.grid()
plt.legend()
plt.show()
"""
