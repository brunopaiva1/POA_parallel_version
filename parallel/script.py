import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nt = 501

fig, ax = plt.subplots()
#fig.set_size_inches(10, 6)

def init():
    ax.clear()

# def update(t):
#     if t % 50 == 0:
#         filename = f"parallel/samples/sample_t{t}.bin"

#         data = np.fromfile(filename, dtype=np.float32)

#         nx, ny, nz = (50, 50, 50)
#         data = data.reshape((nx, ny, nz))

#         plt.imshow(data[:, :, 25], cmap='viridis')
#         plt.title(f'Tempo {t}')

# ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init, repeat=False, interval = 0)
def update(t): 
    if t % 50 == 0:
        filename = f"/home/brunopaiva/POA_parallel_version/parallel/samples/sample_t{t}.bin"

        data = np.fromfile(filename, dtype=np.float32)

        # Determine o tamanho do array original
        total_elements = data.size

        # Defina as dimensões desejadas para a matriz (por exemplo, 50x50x50)
        desired_shape = (50, 50, 50)

        # Calcule a quantidade de elementos necessários para a nova forma
        required_elements = np.prod(desired_shape)

        # Se o número total de elementos não for compatível com a forma desejada,
        # ajuste a forma para corresponder ao número total de elementos
        if total_elements != required_elements:
            data = data[:required_elements]

        # Agora, ajuste a forma para a forma desejada
        data = data.reshape(desired_shape)

ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init, repeat=False, interval = 0)
#plt.colorbar()
plt.show()
#ani.save('wave_animation.mp4', writer='ffmpeg', fps=30)