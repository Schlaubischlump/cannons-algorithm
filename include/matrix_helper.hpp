#include <random>
#include <iostream>
#include <mpi.h>


template <
    typename value_t>
struct matrix {

    int rows;
    int cols;
    value_t *data;

    matrix() {}

    matrix(value_t *data_, int rows_, int cols_) :
      data(data_),
      rows(rows_),
      cols(cols_) {}

    void print() {
      for(int y = 0; y < this->rows; ++y) {
          for(int x = 0; x < this->cols; ++x)
              std::cout <<  this->data[y*this->rows +x] << " \t" ;
          std::cout << std::endl;
      }
      std::cout << std::endl;
    }

    void fill_rnd_int() {
      std::random_device rnd;
      // mt19937 is faster than random_device() which does not need any seeds
      std::mt19937 mt(rnd());
      // Create random numbers in [1, 4294967296)
      std::uniform_int_distribution<value_t> dist(-1, 3);
      for(int y = 0; y < this->rows; y++)
          for(int x = 0; x < this->cols; x++)
              this->data[x + y*cols] = dist(mt);
    }

    void fill_rnd() {
      std::random_device rnd;
      // mt19937 is faster than random_device() which does not need any seeds
      std::mt19937 mt(rnd());
      // Create random numbers in [1, 4294967296)
      std::uniform_real_distribution<value_t> dist(-1, 3);
      for(int y = 0; y < this->rows; y++)
          for(int x = 0; x < this->cols; x++)
              this->data[x + y*cols] = dist(mt);
    }

    void fill_zero() {
      std::fill_n(this->data, this->rows * this->cols, 0);
    }

    value_t &operator()(const int& row, const int& col) {
      return this->data[this->rows*row + col];
    }

    value_t &operator[](const int idx) {
      // get a value from the matrix at the absolut array index
      return this->data[idx];
    }

    bool compare(matrix<value_t> &B) {
      // return true if both matrices are the same, otherwise false
      assert (this->rows == B.rows && this->cols == B.cols);
      for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
          if ((*this)(i, j) != B(i, j)) {
            return false;
          }
        }
      }
      return true;
    }
};
