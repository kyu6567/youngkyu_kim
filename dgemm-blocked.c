const char* dgemm_desc = "team07_hw1";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 60
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) < (b)) ? (b) : (a))

/* This auxiliary subroutine performs a smaller dgemm operation
*  C := C + A * B
* where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* restrict C) {
	static double buff_A[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(0x100)));	// bzero(buff_A, sizeof(buff_A));
	static double buff_C[BLOCK_SIZE] __attribute__((aligned(0x100)));
	bzero(buff_C, sizeof(buff_C));

	/* Buffer full A */
	for (int buff_k = 0; buff_k < K; buff_k++)
		for (int buff_i = 0; buff_i < M; buff_i++) {
			buff_A[buff_i + M * buff_k] = A[buff_i + lda * buff_k];
		}

	for (int j = 0; j < N; ++j) {
		for (int k = 0; k < K; ++k) {
			double buff_B_s = B[k + lda * j];
			for (int i = 0; i < M; i++) {
				buff_C[i] += buff_A[i + k * M] * buff_B_s;
			}
		}
		for (int i = 0; i < M; i++) {
			C[i + lda * j] += buff_C[i];
		}
		bzero(buff_C, sizeof(buff_C));
	}
}

/* This routine performs a dgemm operation
*  C := C + A * B
* where A, B, and C are lda-by-lda matrices stored in column-major format.
* On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
	/* For each block-row of A */
	for (int j = 0; j < lda; j += BLOCK_SIZE)
		/* For each block-column of B */
		for (int k = 0; k < lda; k += BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
			for (int i = 0; i < lda; i += BLOCK_SIZE) {
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M = min(BLOCK_SIZE, lda - i);
				int N = min(BLOCK_SIZE, lda - j);
				int K = min(BLOCK_SIZE, lda - k);

				/* Perform individual block dgemm */
				do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
			}
}
