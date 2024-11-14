#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define M_PI 3.14159265358979323846

const double PI_SQUARE = M_PI * M_PI;
const double PI_SQUARE_FOUR = 4.0 * PI_SQUARE;
const double PI_SQUARE_FIVE = 5.0 * PI_SQUARE;
const double PI_SQUARE_TWELVE = 12.0 * PI_SQUARE;

void generateSource(std::vector<float>& s, float f, float dt, int nt, int thread_count) {
    float t;
#   pragma omp parallel for num_threads(thread_count) \
    default(none) shared(s, PI_SQUARE, f, dt, nt) private(t)
    for (int i = 0; i < nt; i++) {
        t = i * dt;
        s[i] = (1 - PI_SQUARE * f * f * t * t) * exp(-PI_SQUARE * f * f * t * t);
    }    
}

float calculateDEx(const std::vector<float>& previousWavefield, int x, int y, int z, int ny, int nz, float dx) {
             return ((-1.0/12.0)*previousWavefield[(x - 2) * ny * nz + y * nz + z] +
                    (4.0/3.0)*previousWavefield[(x - 1) * ny * nz + y * nz + z] -
                    (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
                    (4.0/3.0)*previousWavefield[(x + 1) * ny * nz + y * nz + z] -
                    (1.0/12.0)*previousWavefield[(x + 2) * ny * nz + y * nz + z]) / (dx * dx);        
}

float calculateDEy(const std::vector<float>& previousWavefield, int x, int y, int z, int ny, int nz, float dy) {
            return ((-1.0/12.0)*previousWavefield[x * ny * nz + (y - 2) * nz + z] +
                    (4.0/3.0)*previousWavefield[x * ny * nz + (y - 1) * nz + z] -
                    (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
                    (4.0/3.0)*previousWavefield[x * ny * nz + (y + 1) * nz + z] -
                    (1.0/12.0)*previousWavefield[x * ny * nz + (y + 2) * nz + z]) / (dy * dy);
}

float calculateDEz(const std::vector<float>& previousWavefield, int x, int y, int z, int ny, int nz, float dz) {
             return ((-1.0/12.0)*previousWavefield[x * ny * nz + y * nz + (z - 2)] +
                    (4.0/3.0)*previousWavefield[x * ny * nz + y * nz + (z - 1)] -
                    (5.0/2.0)*previousWavefield[x * ny * nz + y * nz + z] +
                    (4.0/3.0)*previousWavefield[x * ny * nz + y * nz + (z + 1)] -
                    (1.0/12.0)*previousWavefield[x * ny * nz + y * nz + (z + 2)]) / (dz * dz);

}
void wavePropagation(std::vector<float>& s, float c, float dx, float dy, float dz, float dt,
                    int nx, int ny, int nz, int nt, int xs, int ys, int zs, int thread_count) {
    std::vector<float> previousWavefield(nx * ny * nz, 0.0);
    std::vector<float> nextWavefield(nx * ny * nz, 0.0);
    std::vector<float> u(nx * ny * nz, 0.0);
    float dEx, dEy, dEz;

    for (int t = 0; t < nt; t++) {
#   pragma omp parallel for collapse(3) num_threads(thread_count) \
    default(none) shared(nx, ny, nz, nt, dx, dy, dz, dt, u, previousWavefield, nextWavefield, c, xs, ys, zs, s) \
    private(dEx, dEy, dEz) schedule(dynamic, 2)
        for (int x = 2; x < nx - 2; x++)
        {
            for (int y = 2; y < ny - 2; y++)
            {
                for (int z = 2; z < nz - 2; z++)
                {
                    
            dEx = calculateDEx(previousWavefield, x, y, z, ny, nz, dx);
            dEy = calculateDEy(previousWavefield, x, y, z, ny, nz, dy);
            dEz = calculateDEz(previousWavefield, x, y, z, ny, nz, dz);


            nextWavefield[x * ny * nz + y * nz + z] = c * c * dt * dt * (dEx + dEy + dEz) -
                         previousWavefield[x * ny * nz + y * nz + z] + 2 * u[x * ny * nz + y * nz + z];
              }   
        }
    }

        nextWavefield[xs * ny * nz + ys * nz + zs] -= c * c * dt * dt * s[t];

        std::vector<float> temp = u;
        u = nextWavefield;
        nextWavefield = previousWavefield;
        previousWavefield = temp;
        
         if (t % 50 == 0) {
            
            char filename[50];
            sprintf(filename, "samples/sample_t%d.bin", t); // Cria um nome de arquivo único para cada tempo
            FILE *file = fopen(filename, "wb");
            if (file != NULL) {
                // Escreva os dados de uProximo no arquivo binário
                fwrite(nextWavefield.data(), sizeof(float), nx * ny * nz, file);
                fclose(file);
            } else {
                printf("Erro ao abrir o arquivo para escrita.\n");
            }
        }
}
}
int main(int argc, char* argv[]) {
    double start, end, exetime;

    start = omp_get_wtime();

    int xs = 15, ys = 15, zs = 15;
    float dx = 10, dy = 10, dz = 10;
    float dt = 0.001;
    int nx = 10, ny = 10, nz = 10;
    int nt = 501;
    float f = 10;
    float c = 1500;
    int thread_count;

    thread_count = strtol(argv[1], NULL, 10);
    std::vector<float> s(nt);

    generateSource(s, f, dt, nt, thread_count);
    wavePropagation(s, c, dx, dy, dz, dt, nx, ny, nz, nt, xs, ys, zs, thread_count);

    end = omp_get_wtime();

    exetime = end - start;

    std::cout << "Time: " << exetime << " seconds" << std::endl;

    return 0;
}