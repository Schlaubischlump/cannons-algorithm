#include "../include/matrix_helper.hpp"
#include "../include/cannon.hpp"

// configure datatypes to use
typedef float data_t;


int main(int argc, char* argv[])
{
  // size of matrix
  int n = 1<<10;
  // Init MPI
  MPI_Init(&argc, &argv);
  int num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  int my_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  // make sure that the processors can be aligned in a 2D grid
  const int size = (int)sqrt(num_proc);

  // processes can be arranged in a grid
  assert (size*size == num_proc);

  // placeholder for matrices
  data_t *data_C = nullptr,
         *data_A = nullptr,
         *data_B = nullptr,
         *data_C2 = nullptr;

  if (!my_id) {
    // create matrices to multiply
    data_A = new data_t[n*n];
    fill_rnd(data_A, n);

    data_B = new data_t[n*n];
    fill_rnd(data_B, n);//_int();

    // create target matrix for serial execution
    data_C = new data_t[n*n];
    fill_zero(data_C, n);

    // create target matrix for parallel execution
    data_C2 = new data_t[n*n];
    fill_zero(data_C2, n);
  }

  // serial: multiply matrix A and B
  double start = MPI_Wtime();
  if(!my_id) naive_multiply_add(n, data_A, data_B, data_C);
  double end = MPI_Wtime();

  if(!my_id) {
    printf("# elapsed time (serial_scoring_matrix): %fs\n", end-start);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // parallel: multiply matrix A and B
  start = MPI_Wtime();
  multiply_parallel<data_t>(n, data_A, data_B, data_C2);
  end = MPI_Wtime();

  if(!my_id) {
    if (compare(data_C, data_C2, n))
      std::cout << "Parallel and serial results are different!" << std::endl;
    printf("# elapsed time (parallel_scoring_matrix): %fs\n", end-start);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // cleanup
  if (!my_id) {
    delete data_A;
    delete data_B;
    delete data_C;
  }

  // finished MPI
  MPI_Finalize();

  return 0;
}
