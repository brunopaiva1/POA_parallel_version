import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#número total de tempos
nt = 501

fig, ax = plt.subplots()

#função de inicialização da animação
def init():
    ax.clear()

#função de atualização da animação
def update(t):
    if t % 50 == 0:
        filename = f"/home/brunopaiva/POA_parallel_version/parallel/samples/sample_t{t}.bin"

        data = np.fromfile(filename, dtype=np.float32)

        # tamanho da matriz
        nx, ny, nz = (50, 50, 50)
        data = data.reshape((nx, ny, nz))

        # 15 é a localização em z da fonte
        plt.imshow(data[:, :, 25], cmap='viridis')
        plt.title(f'Tempo {t}')

# Cria a animação
ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init, repeat=False, interval = 0)

# Mostra a animação
plt.show()