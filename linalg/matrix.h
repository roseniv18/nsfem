#ifndef MATRIX_H
#define MATRIX_H

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include "exceptions/exceptions.h"

using std::unique_ptr, std::make_unique, std::cout, std::move,
    std::random_device, std::mt19937, std::uniform_int_distribution,
    std::uniform_real_distribution, std::chrono::high_resolution_clock,
    std::chrono::duration;

enum class MultStrategy {
  Serial,
  ParallelOuter,
  ParallelInner,
  ParallelCollapse,
  Transposed,
  TransposedCollapse,
  Block,
  TransposedBlock,
  CUDA,
  CUDA_row,
  CUDA_col,
  CUDA_tile
};

struct BenchmarkResult {
  duration<double> total;  // full wall time: alloc + H2D + kernel + D2H + free
                           // (or full CPU call)
  duration<double>
      compute;  // kernel-only for CUDA; == total for CPU strategies
};

template <typename T>
class Matrix;

// DECLARE CUDA KERNELS
template <typename T>
void cuda_matmul(const T* A,
                 const T* B,
                 T* C,
                 int rows,
                 int cols,
                 int out_cols,
                 float* kernel_ms = nullptr);

template <typename T>
void cuda_matmul_row(const T* A,
                     const T* B,
                     T* C,
                     int rows,
                     int cols,
                     int out_cols,
                     float* kernel_ms = nullptr);

template <typename T>
void cuda_matmul_col(const T* A,
                     const T* B,
                     T* C,
                     int rows,
                     int cols,
                     int out_cols,
                     float* kernel_ms = nullptr);

template <typename T>
void cuda_matmul_tile(const T* A,
                      const T* B,
                      T* C,
                      int rows,
                      int cols,
                      int out_cols,
                      float* kernel_ms = nullptr);

template <typename T>
Matrix<T> mult(const Matrix<T>& A,
               const Matrix<T>& B,
               MultStrategy strategy,
               float* device_kernel_ms = nullptr);

template <typename T>
class Matrix {
 public:
  //  CONSTRUCTORS
  Matrix();
  Matrix(int rows, int cols);
  Matrix(const T* matr, int rows, int cols);
  //   COPY CONSTRUCTOR
  Matrix(const Matrix<T>& other);
  //   MOVE CONSTRUCTOR
  Matrix(Matrix<T>&& other) noexcept;
  //   DESTRUCTOR
  ~Matrix() = default;

  Matrix<T>& operator+=(const Matrix<T>& m);
  //   Matrix<T>& operator-=(const Matrix<T>& m);
  Matrix<T>& operator*=(const Matrix<T>& m);
  //   Copy assignment operator
  Matrix<T>& operator=(const Matrix<T>& other);
  //   Move assignment operator
  Matrix<T>& operator=(Matrix<T>&& other) noexcept;
  //   Access
  T& operator()(int row, int col);
  const T& operator()(int row, int col) const;
  //   Equality
  bool operator==(const Matrix<T>& other) const;

  Matrix<T>& mult_serial(const Matrix<T>& m);
  Matrix<T>& mult_parallel(const Matrix<T>& m);
  friend Matrix<T> mult<T>(const Matrix<T>& A,
                           const Matrix<T>& B,
                           MultStrategy strategy,
                           float* device_kernel_ms);

  static BenchmarkResult benchmark_mult(int size, MultStrategy strategy);

  static Matrix<T> transpose(const Matrix<T>& m);
  static Matrix<T> augment(const Matrix<T>& A, const std::vector<T>& b);

  int m;  // rows
  int n;  // columns

  static Matrix<T> random(int rows, int cols, int min = 0, int max = 10);
  int size() const;
  void swap_rows(int row1, int row2);
  void row_scale_and_add(int target_row, int source_row, T scalar);
  void print_matrix() const;

 private:
  unique_ptr<T[]> elements;
};

// Declare global operators
template <typename T>
Matrix<T> operator+(Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T> operator*(Matrix<T>& lhs, const Matrix<T>& rhs);

template <typename T>
Matrix<T>::Matrix() : m(1), n(1), elements(nullptr) {}

template <typename T>
Matrix<T>::Matrix(int rows, int cols) : m(rows), n(cols) {
  if ((rows < 1) || (cols < 1))
    throw MatrixSizeIncompatible(
        "Incorrect matrix row or column size on initialization!");

  elements = make_unique<T[]>(m * n);
}

template <typename T>
Matrix<T>::Matrix(const T* matr, int rows, int cols) : m(rows), n(cols) {
  if ((rows < 1) || (cols < 1))
    throw MatrixSizeIncompatible(
        "Incorrect matrix row or column size on initialization!");

  elements = make_unique<T[]>(m * n);

  for (int k = 0; k < m * n; k++)
    elements[k] = matr[k];
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other) : n(other.n), m(other.m) {
  elements = make_unique<T[]>(m * n);

  for (int k = 0; k < m * n; k++)
    elements[k] = other.elements[k];
}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : n(other.n), m(other.m), elements(move(other.elements)) {
  other.n = 0;
  other.m = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& matr) {
  int rows = this->m;
  int cols = this->n;

  if ((rows != matr.m) || (cols != matr.n))
    throw MatrixSizeIncompatible(
        "Matrix sizes are incompatible : Attempted += operation");

  for (int k = 0; k < rows * cols; k++)
    elements[k] += matr.elements[k];

  return *this;
}
template <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& matr) {
  int rows = this->m;
  int cols = this->n;

  if ((cols != matr.m))
    throw MatrixSizeIncompatible(
        "Matrix sizes are incompatible : Attempted *= operation");

  Matrix<T> result(rows, matr.n);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < matr.n; j++) {
      T sum{};
      for (int k = 0; k < cols; k++) {
        sum += (*this)(i, k) * matr(k, j);
      }
      result(i, j) = sum;
    }
  }

  *this = result;
  return *this;
}

template <typename T>
Matrix<T> operator+(Matrix<T>& lhs, const Matrix<T>& rhs) {
  if ((lhs.m != rhs.m) || (lhs.n != rhs.n))
    throw MatrixSizeIncompatible(
        "Matrix sizes are incompatible : Attempted += operation");

  Matrix<T> sum(lhs);
  sum += rhs;
  return sum;
}

template <typename T>
Matrix<T> operator*(Matrix<T>& lhs, const Matrix<T>& rhs) {
  Matrix<T> product(lhs);
  product *= rhs;
  return product;
}

template <typename T>
T& Matrix<T>::operator()(int row, int col) {
  return elements[row * n + col];
}

template <typename T>
const T& Matrix<T>::operator()(int row, int col) const {
  return elements[row * n + col];
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
  if ((m != other.m) || (n != other.n))
    return false;

  for (int i = 0; i < m * n; i++) {
    if (elements[i] != other.elements[i])
      return false;
  }

  return true;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
  if (this != &other) {
    m = other.m;
    n = other.n;
    elements = make_unique<T[]>(m * n);

    for (int k = 0; k < m * n; k++)
      elements[k] = other.elements[k];
  }
  return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept {
  if (this != &other) {
    m = other.m;
    n = other.n;
    elements = move(other.elements);
    other.m = 0;
    other.n = 0;
  }

  return *this;
}

template <typename T>
Matrix<T> mult(const Matrix<T>& A,
               const Matrix<T>& B,
               MultStrategy strategy,
               float* device_kernel_ms) {
  if (A.n != B.m) {
    throw MatrixSizeIncompatible(
        "Matrix sizes are incompatible : Attempted mult operation");
  }
  int rows = A.m;
  int cols = A.n;
  int out_cols = B.n;
  int BS = 32;

  Matrix<T> result(rows, out_cols);

  auto dot = [&](int i, int j) {
    T sum{};

#pragma omp simd reduction(+ : sum)
    for (int k = 0; k < cols; k++) {
      sum += A(i, k) * B(k, j);
    }

    result(i, j) = sum;
  };

  switch (strategy) {
    case MultStrategy::Serial:
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::ParallelOuter:
#pragma omp parallel for
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::ParallelInner:
      for (int i = 0; i < rows; i++)
#pragma omp parallel for
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::ParallelCollapse:
#pragma omp parallel for collapse(2)
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::Transposed:
#pragma omp parallel for
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::TransposedCollapse:
#pragma omp parallel for collapse(2)
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < out_cols; j++)
          dot(i, j);
      break;

    case MultStrategy::Block:
#pragma omp parallel for
      for (int ii = 0; ii < rows; ii += BS) {
        for (int jj = 0; jj < out_cols; jj += BS) {
          for (int kk = 0; kk < cols; kk += BS) {
            for (int i = ii; i < std::min(ii + BS, rows); i++) {
              for (int j = jj; j < std::min(jj + BS, out_cols); j++) {
                T sum = result(i, j);

#pragma omp simd reduction(+ : sum)
                for (int k = kk; k < std::min(kk + BS, cols); k++) {
                  sum += A(i, k) * B(k, j);
                }

                result(i, j) = sum;
              }
            }
          }
        }
      }
      break;

    case MultStrategy::TransposedBlock:
#pragma omp parallel for
      for (int ii = 0; ii < rows; ii += BS) {
        for (int jj = 0; jj < out_cols; jj += BS) {
          for (int kk = 0; kk < cols; kk += BS) {
            for (int i = ii; i < std::min(ii + BS, rows); i++) {
              for (int j = jj; j < std::min(jj + BS, out_cols); j++) {
                T sum = result(i, j);

#pragma omp simd reduction(+ : sum)
                for (int k = kk; k < std::min(kk + BS, cols); k++) {
                  sum += A(i, k) * B(j, k);
                }

                result(i, j) = sum;
              }
            }
          }
        }
      }
      break;

    case MultStrategy::CUDA:
      cuda_matmul(A.elements.get(), B.elements.get(), result.elements.get(),
                  rows, cols, out_cols, device_kernel_ms);
      break;

    case MultStrategy::CUDA_row:
      cuda_matmul_row(A.elements.get(), B.elements.get(), result.elements.get(),
                      rows, cols, out_cols, device_kernel_ms);
      break;

    case MultStrategy::CUDA_col:
      cuda_matmul_col(A.elements.get(), B.elements.get(), result.elements.get(),
                      rows, cols, out_cols, device_kernel_ms);
      break;

    case MultStrategy::CUDA_tile:
      cuda_matmul_tile(A.elements.get(), B.elements.get(),
                       result.elements.get(), rows, cols, out_cols,
                       device_kernel_ms);
      break;
  }

  return result;
}

template <typename T>
Matrix<T>& Matrix<T>::mult_serial(const Matrix<T>& matr) {
  return *this *= matr;
}

template <typename T>
Matrix<T>& Matrix<T>::mult_parallel(const Matrix<T>& matr) {
  int rows = this->m;
  int cols = this->n;

  if ((cols != matr.m))
    throw MatrixSizeIncompatible(
        "Matrix sizes are incompatible : Attempted *= operation");

  Matrix<T> result(rows, matr.n);
  Matrix<T> matr_tr = transpose(matr);

#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < matr.n; j++) {
      T sum{};

#pragma omp simd reduction(+ : sum)
      for (int k = 0; k < cols; k++) {
        sum += (*this)(i, k) * matr(k, j);
      }

      result(i, j) = sum;
    }
  }

  *this = result;
  return *this;
}

template <typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T>& matr) {
  int rows = matr.m;
  int cols = matr.n;

  Matrix<T> newmatr(cols, rows);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      newmatr(j, i) = matr(i, j);
    }
  }

  return newmatr;
}

template <typename T>
static Matrix<T> augment(const Matrix<T>& A, const std::vector<T>& b) {
  if (A.m != b.size())
    throw MatrixSizeIncompatible(
        "Vector size must match rows of A in order to augment");

  Matrix<T> augmented(A.m, A.n + 1);

  for (int i = 0; i < A.m; i++) {
    for (int j = 0; j < A.n; j++) {
      augmented(i, j) = A(i, j);
    }
    augmented(i, A.n) = b[i];
  }

  return augmented;
}

template <typename T>
Matrix<T> Matrix<T>::random(int rows, int cols, int min, int max) {
  Matrix<T> matr(rows, cols);

  static thread_local mt19937 gen(random_device{}());

  if constexpr (std::is_integral_v<T>) {
    uniform_int_distribution<T> dist(min, max);
    for (int i = 0; i < rows * cols; i++)
      matr.elements[i] = dist(gen);

  } else {
    uniform_real_distribution<T> dist(min, max);
    for (int i = 0; i < rows * cols; i++)
      matr.elements[i] = dist(gen);
  }

  return matr;
}

template <typename T>
int Matrix<T>::size() const {
  return n * m;
}

template <typename T>
void Matrix<T>::swap_rows(int row1, int row2) {
  if (row1 == row2)
    return;

  for (int j = 0; j < n; j++) {
    std::swap((*this)(row1, j), (*this)(row2, j));
  }
}

template <typename T>
void Matrix<T>::row_scale_and_add(int target_row, int source_row, T scalar) {
#pragma omp parallel for schedule(static)
  for (int j = 0; j < n; j++) {
    (*this)(target_row, j) += scalar * (*this)(source_row, j);
  }
}

template <typename T>
void Matrix<T>::print_matrix() const {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cout << elements[i * n + j] << " ";
    }
    cout << '\n';
  }
}

template <typename T>
BenchmarkResult Matrix<T>::benchmark_mult(int size, MultStrategy strategy) {
  const Matrix<T> A = Matrix<T>::random(size, size);
  const Matrix<T> B = Matrix<T>::random(size, size);

  const Matrix<T>* B_ptr = &B;
  Matrix<T> B_t(size, size);
  if (strategy == MultStrategy::Transposed ||
      strategy == MultStrategy::TransposedCollapse ||
      strategy == MultStrategy::TransposedBlock) {
    B_t = Matrix<T>::transpose(B);
    B_ptr = &B_t;
  }

  bool is_cuda =
      strategy == MultStrategy::CUDA || strategy == MultStrategy::CUDA_row ||
      strategy == MultStrategy::CUDA_col || strategy == MultStrategy::CUDA_tile;

  // Warm-up: first CUDA call in the process pays context creation + JIT cost.
  if (is_cuda) {
    Matrix<T> warm = mult(A, *B_ptr, strategy);
  }

  float kernel_ms = 0.0f;
  auto start = high_resolution_clock::now();
  Matrix<T> C = mult(A, *B_ptr, strategy, is_cuda ? &kernel_ms : nullptr);
  auto end = high_resolution_clock::now();

  duration<double> total = end - start;
  duration<double> compute =
      is_cuda ? duration<double>(kernel_ms / 1000.0) : total;

  return {total, compute};
}

#endif
