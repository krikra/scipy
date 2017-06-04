#ifndef __CSRS_H__
#define __CSRS_H__

#include <set>
#include <vector>
#include <algorithm>
#include <functional>

#include "util.h"
#include "dense.h"

#include "csr.h"

/*
 * Extract main diagonal of CSR-S matrix A
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[n_col]     - nonzeros
 *
 * Output Arguments:
 *   T  Yx[min(n_row,n_col)] - diagonal entries
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Duplicate entries will be summed.
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + min(n_row,n_col))
 * 
 */
template <class I, class T>
void csrs_diagonal(const I n_row,
                  const I n_col, 
	              const I Ap[], 
	              const I Aj[], 
	              const T Ax[],
	                    T Yx[])
{
    const I N = std::min(n_row, n_col);

    for(I i = 0; i < N; i++){
        const I row = Ap[i];

        T diag = 0;
		  diag += Ax[row];

        Yx[i] = diag;
    }
}


/*
 * Compute B = A for CSR-S matrix A, CSR matrix B
 *
 * Also, with the appropriate arguments can also be used to:
 *   - compute B = A^t for CSR matrix A, CSR matrix B
 *   - compute B = A^t for CSC matrix A, CSC matrix B
 *   - convert CSC->CSR
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *
 * Output Arguments:
 *   I  Bp[n_col+1] - column pointer
 *   I  Bj[nnz(A)]  - row indices
 *   T  Bx[nnz(A)]  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, Bx must be preallocated
 *
 * Note: 
 *   Input:  column indices *are not* assumed to be in sorted order
 *   Input:  the diagonal values are assumed to be non-zero
 *   Output: row indices *will be* in sorted order
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 * 
 */
template <class I, class T>
void csrs_tocsr(const I n_row,
	           const I n_col, 
	           const I Ap[], 
	           const I Aj[], 
	           const T Ax[],
	                 I Bp[],
	                 I Bj[],
	                 T Bx[])
{  
    const I nnz = Ap[n_row];

    //compute number of non-zero entries per column of A 
    std::fill(Bp, Bp + n_row, 0);

    for(I row = 0; row < n_row; row++){
	     Bp[row]++;
	     for(I jj = Ap[row]+1; jj < Ap[row+1]; jj++){
		      Bp[row]++;
		      Bp[Aj[jj]]++;
		  }
	 }

    //cumsum the nnz per column to get Bp[]
    for(I col = 0, cumsum = 0; col < n_col; col++){     
        I temp  = Bp[col];
        Bp[col] = cumsum;
        cumsum += temp;
    }
    Bp[n_col] = nnz; 

    for(I row = 0; row < n_row; row++){
        for(I jj = Ap[row]; jj < Ap[row+1]; jj++){
            I col  = Aj[jj];
            I dest = Bp[col];

            Bj[dest] = row;
            Bx[dest] = Ax[jj];

            Bp[col]++;
        }
    }
    for(I row = 0; row < n_row - 1; row++){
	     std::copy(&Aj[Ap[row]+1], &Aj[Ap[row+1]], &Bj[Bp[row]]);
	     std::copy(&Ax[Ap[row]+1], &Ax[Ap[row+1]], &Bx[Bp[row]]);
    }


    for(I row = 0, last = 0; row <= n_row; row++){
        I temp  = Bp[row] + (Ap[row+1] - (Ap[row]+1));
        Bp[row] = last;
        last    = temp;
    }
}   


/* element-wise binary operations*/
template <class I, class T, class T2>
void csrs_ne_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],      T2 Cx[])
{
    csr_ne_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T, class T2>
void csrs_lt_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],      T2 Cx[])
{
    csr_lt_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T, class T2>
void csrs_gt_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],      T2 Cx[])
{
    csr_gt_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T, class T2>
void csrs_le_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],      T2 Cx[])
{
    csr_le_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T, class T2>
void csrs_ge_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],      T2 Cx[])
{
    csr_ge_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T>
void csrs_elmul_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    csr_elmul_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T>
void csrs_eldiv_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    csr_eldiv_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}


template <class I, class T>
void csrs_plus_csrs(const I n_row, const I n_col, 
                  const I Ap[], const I Aj[], const T Ax[],
                  const I Bp[], const I Bj[], const T Bx[],
                        I Cp[],       I Cj[],       T Cx[])
{
    csr_plus_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T>
void csrs_minus_csrs(const I n_row, const I n_col, 
                   const I Ap[], const I Aj[], const T Ax[],
                   const I Bp[], const I Bj[], const T Bx[],
                         I Cp[],       I Cj[],       T Cx[])
{
    csr_minus_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T>
void csrs_maximum_csrs(const I n_row, const I n_col, 
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    csr_maximum_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}

template <class I, class T>
void csrs_minimum_csrs(const I n_row, const I n_col, 
                     const I Ap[], const I Aj[], const T Ax[],
                     const I Bp[], const I Bj[], const T Bx[],
                           I Cp[],       I Cj[],       T Cx[])
{
    csr_minimum_csr(n_row,n_col,Ap,Aj,Ax,Bp,Bj,Bx,Cp,Cj,Cx);
}


/*
 * Compute Y += A*X for CSR matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_row)
 * 
 */
template <class I, class T>
void csrs_matvec(const I n_row,
	            const I n_col, 
	            const I Ap[], 
	            const I Aj[], 
	            const T Ax[],
	            const T Xx[],
	                  T Yx[])
{
	I n_threads = omp_get_max_threads();
	T *work = (T *)calloc(sizeof(T), n_threads * n_col);

/*
	//diagnal block
	#pragma omp parallel private(diag, jj)
    for(I i = row_start; i < row_end; i++){
	    I diag = Ap[i];
        Yx[i] += Ax[diag] * Xx[i];
        for(I jj = Ap[i]+1; jj < Ap[i+1]; jj++){
            Yx[i]      += Ax[jj] * Xx[Aj[jj]];
            Yx[Aj[jj]] += Ax[jj] * Xx[i];
        }
    }
*/

	//off-diagonal block
	T sum = 0.0;
	#pragma omp parallel private(sum)
	{
	I th = omp_get_thread_num();
	//I row_start = n_row * th / n_threads;
	//I row_end = n_row * (th + 1) / n_threads;

    //for(I i = row_start; i < row_end; i++){
	#pragma omp for
    for(I i = 0; i < n_row; i++){
		sum = 0.0;
        for(I jj = Ap[i]+1; jj < Ap[i+1]; jj++){
            sum                       += Ax[jj] * Xx[Aj[jj]];
            work[Aj[jj] + th * n_col] += Ax[jj] * Xx[i];
        }
	    I diag = Ap[i];
        Yx[i] = Ax[diag] * Xx[i] + sum;
    }

	#pragma omp for
	for(I i = 0; i < n_col; i++){
		sum = 0.0;
		for(I j = 0; j < n_threads; j++){
			sum += work[i + n_col * j];
		}

		Yx[i] += sum;
	}
	}
	
	free(work);
}


/*
 * Compute Y += A*X for CSR matrix A and dense block vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_row            - number of rows in A
 *   I  n_col            - number of columns in A
 *   I  n_vecs           - number of column vectors in X and Y
 *   I  Ap[n_row+1]      - row pointer
 *   I  Aj[nnz(A)]       - column indices
 *   T  Ax[nnz(A)]       - nonzeros
 *   T  Xx[n_col,n_vecs] - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row,n_vecs] - output vector
 *
 */
template <class I, class T>
void csrs_matvecs(const I n_row,
	             const I n_col, 
                 const I n_vecs,
	             const I Ap[], 
	             const I Aj[], 
	             const T Ax[],
	             const T Xx[],
	                   T Yx[])
{
    for(I i = 0; i < n_row; i++){
        T * y_u = Yx + (npy_intp)n_vecs * i;
        const T * x_l = Xx + (npy_intp)n_vecs * i;
        for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
            const I j = Aj[jj];
            const T a = Ax[jj];
            const T * x_u = Xx + (npy_intp)n_vecs * j;
           	T * y_l = Yx + (npy_intp)n_vecs * j;
            axpy(n_vecs, a, x_u, y_u);
            axpy(n_vecs, a, x_l, y_l);
        }
    }
}

#endif
