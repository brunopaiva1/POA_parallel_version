#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

using namespace std;

void generateSource(float *s, float f, float dt, int nt) {
    float t;
    float pi = 3.14;

    for (int i = 0; i < nt; i++){
        t = i * dt;
        s[i] = (1 - 2 * pi * pi * f * f * t * t) * exp(-pi * pi * f * f * t * t);
    }
    
}

void wavePropagation(float *s, float c, float dx, float dy, float dz, float dt,
                    int nx, int ny, int nz, int nt, int xs, int ys, int zs) {
    
    float dEx, dEy, dEz;
    float *uAnterior = (float*) malloc(nx * ny * nz * sizeof(float));
    float *uProximo = (float*) malloc(nx * ny * nz * sizeof(float));
    float *u = (float*) malloc(nx * ny * nz * sizeof(float));

    memset(u, 0, nx * ny * nz * sizeof(float));
    memset(uAnterior, 0, nx * ny * nz * sizeof(float));
    memset(uProximo, 0, nx * ny * nz * sizeof(float));

    for (int t = 0; t < nt; t++) {
        for (int idx = 0; idx < (nx - 4) * (ny - 4) * (nz - 4); idx++) {
            int x = 2 + idx / ((ny - 4) * (nz - 4));
            int y = 2 + (idx / (nz - 4)) % (ny - 4);
            int z = 2 + idx % (nz - 4);

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

            uProximo[x * ny * nz + y * nz + z] = c * c * dt * dt * (dEx + dEy + dEz) - 
                    uAnterior[x * ny * nz + y * nz + z] + 2 * u[x * ny * nz + y * nz + z];
                    
        }


        uProximo[xs * ny * nz + ys * nz + zs] -= c * c * dt * dt * s[t];

        float *temp = u;
        u = uProximo;
        uProximo = uAnterior;
        uAnterior = temp; 
    }
    
    free(uAnterior);
    free(uProximo);
    free(u);
}

int main() {
    clock_t start_t, end_t;
    double total_t;
    int xs = 15, ys = 15, zs = 15;
    float dx = 10, dy = 10, dz = 10;
    float dt = 0.001;
    int nx = 20, ny = 20, nz = 20;
    int nt = 10000;
    float f = 10;
    float c = 1500.0;

    float *s = (float *)malloc(nt * sizeof(float));

    start_t = clock();
    generateSource(s, f, dt, nt);
    wavePropagation(s, c, dx, dy, dz, dt, nx, ny, nz, nt, xs, ys, zs);
    end_t = clock();

    free(s);

    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    cout << "O tempo de execução é: " << total_t << endl;
    return 0;
}