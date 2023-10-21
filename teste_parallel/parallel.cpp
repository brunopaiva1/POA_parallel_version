#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

#define M_PI 3.14159265358979323846

const double PI_SQUARE = M_PI * M_PI;
const double PI_SQUARE_FOUR = 4.0 * PI_SQUARE;
const double PI_SQUARE_FIVE = 5.0 * PI_SQUARE;
const double PI_SQUARE_TWELVE = 12.0 * PI_SQUARE;

void generateSource(std::vector<float>& s, float f, float dt, int nt, int thread_count) {
    #pragma omp parallel for
    for (int i = 0; i < nt; i++) {
        float t = i * dt;
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

    for (int t = 0; t < nt; t++) {
        #pragma omp parallel for num_threads(thread_count)
        for (int idx = 0; idx < (nx - 4) * (ny - 4) * (nz - 4); idx++) {
            int x = 2 + idx / ((ny - 4) * (nz - 4));
            int y = 2 + (idx / (nz - 4)) % (ny - 4);
            int z = 2 + idx % (nz - 4);

            float dEx = calculateDEx(previousWavefield, x, y, z, ny, nz, dx);
            float dEy = calculateDEy(previousWavefield, x, y, z, ny, nz, dy);
            float dEz = calculateDEz(previousWavefield, x, y, z, ny, nz, dz);


            nextWavefield[x * ny * nz + y * nz + z] = c * c * dt * dt * (dEx + dEy + dEz) - previousWavefield[x * ny * nz + y * nz + z] + 2 * u[x * ny * nz + y * nz + z];
        }

        nextWavefield[xs * ny * nz + ys * nz + zs] -= c * c * dt * dt * s[t];

        std::vector<float> temp = u;
        u = nextWavefield;
        nextWavefield = previousWavefield;
        previousWavefield = temp;
    }
    
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int xs = 15, ys = 15, zs = 15;
    float dx = 10, dy = 10, dz = 10;
    float dt = 0.001;
    int nx = 20, ny = 20, nz = 20;
    int nt = 10000;
    float f = 10;
    float c = 1500.0;
    int thread_count;

    thread_count = strtol(argv[1], NULL, 10);
    std::vector<float> s(nt);

    generateSource(s, f, dt, nt, thread_count);
    wavePropagation(s, c, dx, dy, dz, dt, nx, ny, nz, nt, xs, ys, zs, thread_count);

    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "O tempo de execução é: " << execution_time << " ms" << std::endl;

    return 0;
}