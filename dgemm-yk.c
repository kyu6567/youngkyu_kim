const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define ALIGN __attribute__ ((aligned (32)))

#include <immintrin.h>

/* This auxiliary subroutine performs a smaller dgemm operation
*  C := C + A * B
* where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	double* temp;

	/* Alligning */
	double ALIGN buff_A[BLOCK_SIZE * BLOCK_SIZE];
	double ALIGN buff_B[BLOCK_SIZE * BLOCK_SIZE];

	/* Padding */
	if (M < BLOCK_SIZE || K < BLOCK_SIZE)
		bzero(buff_A, sizeof(buff_A));
	if (K < BLOCK_SIZE || N < BLOCK_SIZE)
		bzero(buff_B, sizeof(buff_B));
	
	/* Load data */
	int m = 0;
	for (int i = 0; i < M; i++)
		for (int k = 0; k < K; k++)
			buff_A[m++] = A[i + lda*k];
	m = 0;
	for (int j = 0; j < N; j++)
		for (int k = 0; k < K; k++)
			buff_B[m++] = B[k + lda*j];

	/* Operation */
	for (inti = 0; i < 32; ++i)
		for (int j = 0; j < 32; ++j)
		{
			double cij = C[i + lda*j];
			for (int k = 0; k < 2; ++k)
			{
				ymm0 = _mm256_load_pd(buff_A + 16 * k + 32 * i);
				ymm1 = _mm256_load_pd(buff_A + 16 * k + 4 + 32 * i);
				ymm2 = _mm256_load_pd(buff_A + 16 * k + 8 +32 * i);
				ymm3 = _mm256_load_pd(buff_A + 16 * k + 12 + 32 * i);
				ymm4 = _mm256_load_pd(buff_B + 16 * k + 32 * j);
				ymm5 = _mm256_load_pd(buff_B + 16 * k + 4 + 32 * j);
				ymm6 = _mm256_load_pd(buff_B + 16 * k + 8 + 32 * j);
				ymm7 = _mm256_load_pd(buff_B + 16 * k + 12 + 32 * j);

				ymm0 = _mm256_mul_pd(ymm0, ymm4);
				ymm1 = _mm256_mul_pd(ymm1, ymm5);
				ymm2 = _mm256_mul_pd(ymm2, ymm6);
				ymm3 = _mm256_mul_pd(ymm3, ymm7);

				ymm0 = _mm256_add_pd(ymm0, ymm1);
				ymm2 = _mm256_add_pd(ymm2, ymm3);
				ymm0 = _mm256_add_pd(ymm0, ymm2);

				_mm256_store_pd(temp, ymm0);
				for (int m = 0; m < 4; ++m)
					cij += temp[m];
			}
			C[i + lda*j] = cij;
		}
}

/* This routine performs a dgemm operation
*  C := C + A * B
* where A, B, and C are lda-by-lda matrices stored in column-major format.
* On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C)
{
	/* For each block-row of A */
	for (int j = 0; j < lda; j += BLOCK_SIZE)
		/* For each block-column of B */
		for (int k = 0; k < lda; k += BLOCK_SIZE)
			/* Accumulate block dgemms into block of C */
			for (int i = 0; i < lda; i += BLOCK_SIZE)
			{
				/* Correct block dimensions if block "goes off edge of" the matrix */
				int M = min(BLOCK_SIZE, lda - i);
				int N = min(BLOCK_SIZE, lda - j);
				int K = min(BLOCK_SIZE, lda - k);

				/* Perform individual block dgemm */
				do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
}
