#include <chrono>
#include <cstddef>
#include <iostream>
#include <utility>

#include "mlas.h"

void ref_sgemm(float *A, float *B, float *C, size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0;

      for (size_t p = 0; p < k; ++p) {
        sum += A[i * k + p] * B[p * n + j];
      }

      C[i * n + j] = sum;
    }
  }
}

std::pair<float, float> GetRange(float *x, int n) {
  float min_value = x[0];
  float max_value = x[0];

  for (int i = 0; i < n; ++i) {
    float value = x[i];

    if (value < min_value) min_value = value;
    if (value > max_value) max_value = value;
  }

  return {min_value, max_value};
}

float GetMaxDiff(float *x, float *y, int n) {
  float max_diff = 0;

  for (int i = 0; i < n; ++i) {
    float diff = x[i] - y[i];

    if (diff < 0) diff = -diff;
    if (diff > max_diff) max_diff = diff;
  }

  return max_diff;
}

int main(int argc, char *argv[]) {
  size_t m = 1024;
  size_t n = 1024;
  size_t k = 1024;

  float *A = new float[m * k];
  float *B = new float[k * n];
  float *C0 = new float[m * n];
  float *C1 = new float[m * n];

  // init A, B
  for (int i = 0; i < m * k; ++i) A[i] = (float)i / (m * k);
  for (int i = 0; i < k * n; ++i) B[i] = (float)i / (k * n);

  float alpha = 1.0f;
  float beta = 0.0f;

  size_t lda = k;
  size_t ldb = n;
  size_t ldc = n;

  {
    ref_sgemm(A, B, C0, m, n, k);
    MlasGemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C1, ldc, nullptr);

    size_t csize = m * n;

    auto r0 = GetRange(C0, csize);
    auto r1 = GetRange(C1, csize);
    auto d = GetMaxDiff(C0, C1, csize);

    std::cout << "ref  : " << r0.first << ", " << r0.second << std::endl;
    std::cout << "cpp  : " << r1.first << ", " << r1.second << std::endl;
    std::cout << "diff : " << d << "\n" << std::endl;
  }

  constexpr int kRepeats = 10;

  float time[kRepeats] = {0};

  for (int i = 0; i < kRepeats; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();

    MlasGemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C1, ldc, nullptr);

    auto t2 = std::chrono::high_resolution_clock::now();
    time[i] = std::chrono::duration<float, std::micro>(t2 - t1).count() * 0.001f;
  }

  std::cout << "benchmark sgemm (ms)" << std::endl;

  for (int i = 0; i < kRepeats - 1; ++i) std::cout << time[i] << ", ";
  std::cout << time[kRepeats - 1] << std::endl;

  delete[] A;
  delete[] B;
  delete[] C0;
  delete[] C1;

  return 0;
}
