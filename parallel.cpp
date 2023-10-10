#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

// Função para inicializar a fonte s
void initializeSource(float *s, float f, float dt, int nt, int thread_count) {
    float t;
    float pi = 3.14;

#   pragma omp parallel for num_threads(thread_count)\
    default(none) shared(s, pi, f, dt, nt) private(t)
    for (int i = 0; i < nt; i++)
    {
        t = i * dt;
        s[i] = (1 - 2 * pi * pi * f * f * t * t) * exp(-pi * pi * f * f * t * t);
    }
    
}

// Função para propagar a onda acústica usando diferenças finitas
void propagateWave(float *s, float c, float dx, float dy, float dz, float dt,
                    int nx, int ny, int nz, int nt, int xs, int ys, int zs, int thread_count) {
    
    float dEx, dEy, dEz;
    float *uAnterior = malloc(nx * ny * nz * sizeof(float));
    float *uProximo = malloc(nx * ny * nz * sizeof(float));
    float *u = malloc(nx * ny * nz * sizeof(float));

    memset(u, 0, nx * ny * nz * sizeof(float));
    memset(uAnterior, 0, nx * ny * nz * sizeof(float));
    memset(uProximo, 0, nx * ny * nz * sizeof(float));

    for (int t = 0; t < nt; t++)
    {
#       pragma omp parallel for collapse(3) num_threads(thread_count)\
        default(none) shared(nx, ny, nz, nt, dx, dy, dz, dt, u, uAnterior, uProximo, c, xs, ys, zs, s) private(dEx, dEy, dEz)
        for (int x = 2; x < nx - 2; x++)
        {
            for (int y = 2; y < ny - 2; y++)
            {
                for (int z = 2; z < nz - 2; z++)
                {

                    dEx = ((-1.0/12.0)*uAnterior[(x - 2) * ny * nz + y * nz + z] +
                            (4.0/3.0)*uAnterior[(x - 1) * ny * nz + y * nz + z] -
                            (5.0/2.0)*uAnterior[x * ny * nz + y * nz + z] +
                            (4.0/3.0)*uAnterior[(x + 1) * ny * nz + y * nz + z] -
                            (1.0/12.0)*uAnterior[(x + 2) * ny * nz + y * nz + z]) / (dx * dx);
                    
                    dEy = ((-1.0/12.0)*uAnterior[x * ny * nz + (y - 2) * nz + z] +
                            (4.0/3.0)*uAnterior[x * ny * nz + (y - 1) * nz + z] -
                            (5.0/2.0)*uAnterior[x * ny * nz + y * nz + z] +
                            (4.0/3.0)*uAnterior[x * ny * nz + (y + 1) * nz + z] -
                            (1.0/12.0)*uAnterior[x * ny * nz + (y + 2) * nz + z]) / (dy * dy);

                    dEz = ((-1.0/12.0)*uAnterior[x * ny * nz + y * nz + (z - 2)] +
                            (4.0/3.0)*uAnterior[x * ny * nz + y * nz + (z - 1)] -
                            (5.0/2.0)*uAnterior[x * ny * nz + y * nz + z] +
                            (4.0/3.0)*uAnterior[x * ny * nz + y * nz + (z + 1)] -
                            (1.0/12.0)*uAnterior[x * ny * nz + y * nz + (z + 2)]) / (dz * dz);

                    uProximo[x * ny * nz + y * nz + z] = c * c * dt * dt * (dEx + dEy + dEz) - uAnterior[x * ny * nz + y * nz + z] + 2 * u[x * ny * nz + y * nz + z];
                    
                }
            } 
        }

        //uProximo[xs][ys][zs] -= c*c * dt*dt * s[t];
        uProximo[xs * ny * nz + ys * nz + zs] -= c * c * dt * dt * s[t];

        float *temp = u;
        u = uProximo;
        uProximo = uAnterior;
        uAnterior = temp;

        if (t % 50 == 0)
        {
            
            char filename[50];
            sprintf(filename, "samples/sample_t%d.bin", t); // Cria um nome de arquivo único para cada tempo
            FILE *file = fopen(filename, "wb");
            if (file != NULL) {
                // Escreva os dados de uProximo no arquivo binário
                fwrite(uProximo, sizeof(float), nx * ny * nz, file);
                fclose(file);
            } else {
                printf("Erro ao abrir o arquivo para escrita.\n");
            }
        }
        

    }
    
    free(uAnterior);
    free(uProximo);
    
}

int main(int argc, char* argv[]) {
    // Parâmetros de entrada
    int xs = 15, ys = 15, zs = 15;  // Posição da fonte
    float dx = 10, dy = 10, dz = 10;  // Resolução espacial
    float dt = 0.001;         // Passo de tempo
    int nx = 50, ny = 50, nz = 50;   // Dimensões da malha tridimensional
    int nt = 10000;           // Número de passos de tempo
    float f = 10;  // Frequência de pico da fonte
    int c = 1500; //Velocidade de propagação da onda no meio
    int thread_count; //Número de threads

    thread_count = strtol(argv[1], NULL, 10);

    float *s = (float *)malloc(nt * sizeof(float));

    // Inicialize a fonte s
    initializeSource(s, f, dt, nt, thread_count);

    // Propague a onda acústica
    propagateWave(s, c, dx, dy, dz, dt, nx, ny, nz, nt, xs, ys, zs, thread_count);

    // Libere a memória alocada
    free(s);

    return 0;
}