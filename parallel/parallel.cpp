#include <iostream>
#include <math.h>
#include <cmath>
#include <omp.h>

#define M_PI 3.14159265358979323846

const double PI_SQUARE = M_PI * M_PI;
const double PI_SQUARE_FOUR = 4.0 * PI_SQUARE;
const double PI_SQUARE_FIVE = 5.0 * PI_SQUARE;
const double PI_SQUARE_TWELVE = 12.0 * PI_SQUARE;

void generateSource(float* s, float f, float dt, int nt, int thread_count) {
    float t;
#   pragma omp parallel for num_threads(thread_count) \
    default(none) shared(s, PI_SQUARE, f, dt, nt) private(t)
    for (int i = 0; i < nt; i++) {
        t = i * dt;
        s[i] = (1 - 2 * PI_SQUARE * f * f * t * t) * exp(-PI_SQUARE * f * f * t * t);
    }    
}

float calcX(float* previousWavefield, int x, int y, int z, int ny, int nz, float dx) {
    return ((-1.0/12.0)*previousWavefield[(x - 2) * ny * nz + y * nz + z] +
            (4.0/3.0)*previousWavefield[(x - 1) * ny * nz + y * nz + z] -
            (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
            (4.0/3.0)*previousWavefield[(x + 1) * ny * nz + y * nz + z] -
            (1.0/12.0)*previousWavefield[(x + 2) * ny * nz + y * nz + z]) / (dx * dx);        
}

float calcY(float* previousWavefield, int x, int y, int z, int ny, int nz, float dy) {
    return ((-1.0/12.0)*previousWavefield[x * ny * nz + (y - 2) * nz + z] +
            (4.0/3.0)*previousWavefield[x * ny * nz + (y - 1) * nz + z] -
            (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
            (4.0/3.0)*previousWavefield[x * ny * nz + (y + 1) * nz + z] -
            (1.0/12.0)*previousWavefield[x * ny * nz + (y + 2) * nz + z]) / (dy * dy);
}

float calcZ(float* previousWavefield, int x, int y, int z, int ny, int nz, float dz) {
    return ((-1.0/12.0)*previousWavefield[x * ny * nz + y * nz + (z - 2)] +
            (4.0/3.0)*previousWavefield[x * ny * nz + y * nz + (z - 1)] -
            (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
            (4.0/3.0)*previousWavefield[x * ny * nz + y * nz + (z + 1)] -
            (1.0/12.0)*previousWavefield[x * ny * nz + y * nz + (z + 2)]) / (dz * dz);
}

void wavePropagation(float* s, float c, float dx, float dy, float dz, float dt,
                    int nx, int ny, int nz, int nt, int xs, int ys, int zs, int thread_count) {
    float* previousWavefield = new float[nx * ny * nz]();
    float* nextWavefield = new float[nx * ny * nz]();
    float* u = new float[nx * ny * nz]();
    float dEx, dEy, dEz;

    for (int t = 0; t < nt; t++) {
#   pragma omp parallel for collapse(3) num_threads(thread_count) \
    default(none) shared(nx, ny, nz, nt, dx, dy, dz, dt, u, previousWavefield, nextWavefield, c, xs, ys, zs, s) \
    private(dEx, dEy, dEz) schedule(dynamic, 2)
        for (int x = 2; x < nx - 2; x++) {
            for (int y = 2; y < ny - 2; y++) {
                for (int z = 2; z < nz - 2; z++) {
                    dEx = calcX(previousWavefield, x, y, z, ny, nz, dx);
                    dEy = calcY(previousWavefield, x, y, z, ny, nz, dy);
                    dEz = calcZ(previousWavefield, x, y, z, ny, nz, dz);

                    nextWavefield[x * ny * nz + y * nz + z] = c * c * dt * dt * (dEx + dEy + dEz) -
                                                             previousWavefield[x * ny * nz + y * nz + z] + 2 * u[x * ny * nz + y * nz + z];
                }
            }
        }

        nextWavefield[xs * ny * nz + ys * nz + zs] -= c * c * dt * dt * s[t];

        float* temp = u;
        u = nextWavefield;
        nextWavefield = previousWavefield;
        previousWavefield = temp;

        if (t % 50 == 0) {
            char filename[100];
            sprintf(filename, "/home/brunopaiva/POA_parallel_version/parallel/samples/sample_t%d.bin", t); 
            FILE *file = fopen(filename, "wb");
            if (file != NULL) {
                fwrite(nextWavefield, sizeof(float), nx * ny * nz, file);
                fclose(file);
            } else {
                printf("Erro ao abrir o arquivo para escrita.\n");
            }
        }
    }

    delete[] previousWavefield;
    delete[] nextWavefield;
    delete[] u;
}

int main(int argc, char* argv[]) {
    double start, end, exetime;

    start = omp_get_wtime();

    int xs = 25, ys = 25, zs = 25;
    float dx = 10, dy = 10, dz = 10;
    float dt = 0.001;
    int nx = 50, ny = 50, nz = 50;
    int nt = 501;
    float f = 10;
    float c = 1500;
    int thread_count;

    thread_count = strtol(argv[1], NULL, 10);
    float* s = new float[nt];

    generateSource(s, f, dt, nt, thread_count);
    wavePropagation(s, c, dx, dy, dz, dt, nx, ny, nz, nt, xs, ys, zs, thread_count);

    delete[] s;

    end = omp_get_wtime();

    exetime = end - start;

    std::cout << "Time: " << exetime << " seconds" << std::endl;

    return 0;
}
