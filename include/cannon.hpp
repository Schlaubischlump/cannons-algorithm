#ifndef CANNON_HPP
#define CANNON_HPP

#define MPI_VALUE_TYPE MPI_INT


template <
    typename value_t>
void naive_multiply_add(int size, value_t *A, value_t *B, value_t *C)
{
  // simple matrix multiplication C = C + A*B
  //#pragma omp parallel for schedule(dynamic) collapse(2)
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      value_t s = 0;
      for (int k = 0; k < size; ++k) {
        s += A[i*size+k] * B[k*size+j];
      }
      C[i*size+j] += s;
    }
  }
}


template <
    typename value_t>
void multiply_parallel(int N,
                       value_t *A,
                       value_t *B,
                       value_t *C)
{
  // save the id of the current process and the total number of processes
  int num_proc, my_id;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  // amount of blocks in one row / column of the matrix
  const int grid_size = (int)sqrt(num_proc);
  // amount of columns / rows in one matrix block (pad with zeros)
  const int extra_space = N%grid_size ? grid_size - (N % grid_size) : 0;
  const int block_size = (N + extra_space)/grid_size;
  // number of elements in one matrix block
  const int local_matrix_size = block_size*block_size;


  // create 2D sqrt(p) x sqrt(p) grid communicator
  const int periods[2] = {1, 1}; // periodic in both dimensions
  const int dims[2] = {grid_size, grid_size}; // size of each dimension
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

  // local matrices
  value_t *local_A = new value_t[local_matrix_size];
  value_t *local_B = new value_t[local_matrix_size];
  value_t *local_C = new value_t[local_matrix_size];
  // fill matrices with zeros before adding values to it
  std::fill_n(local_C, local_matrix_size, 0);
  std::fill_n(local_A, local_matrix_size, 0);
  std::fill_n(local_B, local_matrix_size, 0);

  // get coordinates of process inside the 2D grid
  int coords[2] = {0, 0};
  MPI_Cart_coords(grid_comm, my_id, grid_size, coords);
  // all indices smaller than this number need to send less data
  const int adjust_indices = (grid_size - (N % grid_size)) % grid_size;
  // datatypes and counts + displacements for send and receive
  MPI_Datatype TMP,
    BLOCK[num_proc], // send data from A or B to local matrix
    LOCAL_BLOCK[num_proc], // Receive local_A or local_B from A or B
    GATHER_LOCAL[num_proc]; // Gather local_C in C
  int *displs = new int[num_proc];
  int row_offset = 0, col_offset = 0;

  for (int row = 0; row < grid_size; ++row) {
    col_offset = 0;
    const int off_row = (row < adjust_indices);
    for (int col = 0; col < grid_size; ++col) {
      // ignore padding for first columns when sending the data
      const int off_col = (col < adjust_indices);
      const int p_id = row*grid_size+col;

      // create datatype to send submatrices (BLOCK)
      MPI_Type_vector(block_size-off_row, block_size-off_col, N, MPI_VALUE_TYPE, &TMP);
      MPI_Type_create_resized(TMP , 0, sizeof(value_t), &BLOCK[p_id]);
      MPI_Type_commit(&BLOCK[p_id]);

      // create datatype to receive submatrices (LOCAL_BLOCK)
      MPI_Type_vector(block_size-off_row, block_size-off_col, block_size, MPI_VALUE_TYPE, &TMP);
      MPI_Type_create_resized(TMP , 0, sizeof(value_t), &LOCAL_BLOCK[p_id]);
      MPI_Type_commit(&LOCAL_BLOCK[p_id]);

      // create datatype to receive submatrices in C (GATHER_LOCAL)
      MPI_Type_vector(block_size-off_row, block_size-off_col, N, MPI_VALUE_TYPE, &TMP);
      MPI_Type_create_resized(TMP , 0, sizeof(value_t), &GATHER_LOCAL[p_id]);
      MPI_Type_commit(&GATHER_LOCAL[p_id]);

      // displacement for send operation
      displs[p_id] = row_offset + col_offset;
      // increase column offset
      col_offset += block_size-off_col;
    }
    // increase row offset
    row_offset += N*(block_size-off_row);
  }

  // scatter data so that each process (i,j) has the entries for the
  // matrix block at index (i, j)
  MPI_Request req;
  if (!my_id) {
    for (int i = 0; i < num_proc; ++i) {
      MPI_Isend(A+displs[i], 1, BLOCK[i], i, 0, grid_comm, &req);
      MPI_Isend(B+displs[i], 1, BLOCK[i], i, 1, grid_comm, &req);
    }
  }
  MPI_Recv(local_A, 1, LOCAL_BLOCK[my_id], 0, 0, grid_comm, MPI_STATUS_IGNORE);
  MPI_Recv(local_B, 1, LOCAL_BLOCK[my_id], 0, 1, grid_comm, MPI_STATUS_IGNORE);

  // wait until all data is send
  MPI_Barrier(grid_comm);
  /*
  After inital scatter:
  Example blocks for matrix A or B (the numbers denote the process ids):
  0,0  |  0, 1  |  0, 2
  1,0  |  1, 1  |  1, 2
  2,0  |  2, 1  |  2, 2
  */

  // ids of neighbours
  int right=0, left=0, down=0, up=0;
  // shift A based on the row and B based on the current column
  MPI_Cart_shift(grid_comm, 1, coords[0], &left, &right);
  MPI_Cart_shift(grid_comm, 0, coords[1], &up, &down);
  MPI_Sendrecv_replace(local_A, local_matrix_size, MPI_VALUE_TYPE, left,
    0, right, 0, grid_comm, MPI_STATUS_IGNORE);
  MPI_Sendrecv_replace(local_B, local_matrix_size, MPI_VALUE_TYPE, up,
    0, down, 0, grid_comm, MPI_STATUS_IGNORE);

  /*
  After inital shift:
  Example blocks for matrix A:       Example blocks for matrix B:
  0,0  |  0, 1  |  0, 2              0,0  |  1, 1  |  2, 2
  1,1  |  1, 2  |  1, 0              1,0  |  2, 1  |  0, 2
  2,2  |  2, 0  |  2, 1              2,0  |  0, 1  |  1, 2
  */

  // multiply and add values to C = C + A*B
  naive_multiply_add(block_size, local_A, local_B, local_C);

  // shift values of A/B to left/up in a loop
  for(int i = 1; i < grid_size; ++i) {
     MPI_Cart_shift(grid_comm, 1, 1, &left,&right);
     MPI_Cart_shift(grid_comm, 0, 1, &up,&down);
     MPI_Sendrecv_replace(local_A, local_matrix_size, MPI_VALUE_TYPE, left,
       0, right, 0, grid_comm, MPI_STATUS_IGNORE);
     MPI_Sendrecv_replace(local_B, local_matrix_size, MPI_VALUE_TYPE, up,
       0, down, 0, grid_comm, MPI_STATUS_IGNORE);
     naive_multiply_add(block_size, local_A, local_B, local_C);
  }

  // gather values in C for final result
  MPI_Isend(local_C, 1, LOCAL_BLOCK[my_id], 0, 0, grid_comm, &req);
  if (!my_id) {
    for (int i = 0; i < num_proc; ++i) {
      MPI_Recv(C+displs[i], 1, GATHER_LOCAL[i], i, 0, grid_comm,
        MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(grid_comm);

  // cleanup
  delete[] displs;
  delete[] local_A;
  delete[] local_B;
  delete[] local_C;
}
#endif // CANNON_HPP
