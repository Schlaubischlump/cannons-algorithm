/*
Helper class for serial execution.
*/
#ifndef MATRIX_HELPER_HPP
#define MATRIX_HELPER_HPP

#include <random>
#include <iostream>
#include <mpi.h>


template <
    typename value_t>
void print_matrix(value_t A, int N) {
  for(int x = 0; x < N; ++x) {
      for(int y = 0; y < N; ++y)
          std::cout <<  A[x*N +y] << " \t" ;
      std::cout << std::endl;
  }
  std::cout << std::endl;
}


template <
    typename value_t>
void fill_rnd(value_t *A, int N) {
  std::random_device rnd;
  // mt19937 is faster than random_device() which does not need any seeds
  std::mt19937 mt(rnd());
  // Create random numbers in [1, 4294967296)
  std::uniform_real_distribution<value_t> dist(-1, 3);
  for(int x = 0; x < N; ++x)
      for(int y = 0; y < N; ++y)
          A[x*N + y] = dist(mt);
}


template <
    typename value_t>
void fill_rnd_int(value_t *A, int N) {
  std::random_device rnd;
  // mt19937 is faster than random_device() which does not need any seeds
  std::mt19937 mt(rnd());
  // Create random numbers in [1, 4294967296)
  std::uniform_int_distribution<value_t> dist(-1, 3);
  for(int x = 0; x < N; ++x)
      for(int y = 0; y < N; ++y)
          A[x*N + y] = dist(mt);
}


template <
    typename value_t>
void fill_zero(value_t *A, int N) {
  std::fill_n(A, N*N, 0);
}


template <
    typename value_t>
bool compare(value_t *A, value_t *B, int N) {
  // return true if both matrices are the same, otherwise false
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (A[i*N+j] != B[i*N+j]) {
        return false;
      }
    }
  }
  return true;
}
#endif // MATRIX_HELPER_HPP
