extern "C" {
  #include <cholmod.h>
}

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

/* add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
int add_entry (cholmod_triplet* T, int i, int j, double val , cholmod_common* c)
{
    //printf("adding entry %d,%d %f\n",i,j,val);
    if ((!T) || i < 0 || j < 0) return (0) ;     /* check inputs */
    if( (T->nnz >= T->nzmax) && 
        !cholmod_reallocate_triplet (2*(T->nzmax),T,c)
    ) 
        return (0) ;
    if (T->x) 
    {
        double* xx = (double*) T->x;
        int* ii = (int*) T->i;
        int* jj = (int*) T->j;
        xx [T->nnz] = val;
        ii [T->nnz] = i ;
        jj [T->nnz++] = j ;
        T->nrow = MAX (T->nrow, i+1) ;
        T->ncol = MAX (T->ncol, j+1) ;
        return (1) ;
    }
    else
    {
        return 0; 
    }
}


cholmod_triplet* create_cholmod_triplet(int nrows, int ncols, int stype, cholmod_common* c)
{
	// stype = 0: cholmod_triplet is "unsymmetric": 
	//			  use both upper and lower triangular parts
	// stype > 0: matrix is square and symmetric. 
	//			  use the lower triangular part to store non-zero entries

        int nzmax = 2*ncols; 
        
        int i = 0; 
        
	cholmod_triplet* triplet_matrix = cholmod_allocate_triplet(nrows, ncols, nzmax, stype, CHOLMOD_REAL, c);

        // fill dyagonal 
        for (i=0; i<ncols; ++i) 
        {
            add_entry(triplet_matrix, i, i, 1., c);
        }
        
        // fill first band above and first band below dyagonal 
        for (i=0; i<ncols-1; ++i) 
        {
            add_entry(triplet_matrix, i, i+1, 2.0 ,c);
        }
        
        // fill first band above and first band below dyagonal 
        for (i=0; i<ncols-2; ++i) 
        {
            add_entry(triplet_matrix, i, i+2, 3.0 ,c);
        }
        
	return triplet_matrix;
}


cholmod_sparse* create_cholmod_sparse(int nrows, int ncols, int stype, cholmod_common* c)
{
    cholmod_triplet* triplet_matrix = create_cholmod_triplet(nrows, ncols, stype, c);
    if (triplet_matrix == 0) return 0; // empty matrix

    cholmod_sparse* output = cholmod_triplet_to_sparse(triplet_matrix, (int)(triplet_matrix->nnz), c);

    cholmod_free_triplet(&triplet_matrix, c);

    return output;
}

int main (void)
{
    cholmod_sparse *A ;
    cholmod_dense *x, *b, *r ;
    cholmod_factor *L ;
    double one [2] = {1,0}, m1 [2] = {-1,0} ;	    /* basic scalars */
    cholmod_common c ;
    
    cholmod_start (&c) ;			    /* start CHOLMOD */
    
    //A = cholmod_read_sparse (stdin, &c) ;	    /* read in a matrix */
    int nrows = 1000; 
    int ncols = nrows;
    int stype = 1; 
/*
    * 0:  matrix is "unsymmetric": use both upper and lower triangular parts
	*     (the matrix may actually be symmetric in pattern and value, but
	*     both parts are explicitly stored and used).  May be square or
	*     rectangular.
	* >0: matrix is square and symmetric.  Entries in the lower triangular
	*     part are transposed and added to the upper triangular part when
	*     the matrix is converted to cholmod_sparse form.
	* <0: matrix is square and symmetric.  Entries in the upper triangular
	*     part are transposed and added to the lower triangular part when
	*     the matrix is converted to cholmod_sparse form.
*/
    A = create_cholmod_sparse(nrows,ncols,stype, &c);
    
    cholmod_print_sparse (A, "A", &c) ;		/* print the matrix */
    if (A == NULL || A->stype == 0)		    /* A must be symmetric */
    {
	cholmod_free_sparse (&A, &c) ;
	cholmod_finish (&c) ;
	return (0) ;
    }
    b = cholmod_ones (A->nrow, 1, A->xtype, &c) ;   /* b = ones(n,1) */
    L = cholmod_analyze (A, &c) ;		        /* analyze */
    cholmod_factorize (A, L, &c) ;		        /* factorize */
    x = cholmod_solve (CHOLMOD_A, L, b, &c) ;	/* solve Ax=b */
    r = cholmod_copy_dense (b, &c) ;		    /* r = b */
    cholmod_sdmult (A, 0, m1, one, x, r, &c) ;	/* r = r-Ax */
    printf ("norm(b-Ax) %8.1e\n",
	    cholmod_norm_dense (r, 0, &c)) ;	    /* print norm(r) */
    cholmod_free_factor (&L, &c) ;		        /* free matrices */
    cholmod_free_sparse (&A, &c) ;
    cholmod_free_dense (&r, &c) ;
    cholmod_free_dense (&x, &c) ;
    cholmod_free_dense (&b, &c) ;
    cholmod_finish (&c) ;			    /* finish CHOLMOD */
    return (0) ;
}
