#include "sparse/CholmodDenseMatrix.h"
#include "sparse/CholmodDenseVector.h"

int main (void)
{
    cholmod_common c ;
    cholmod_start (&c) ;			    /* start CHOLMOD */
    
    
    cholmod_finish (&c) ;			    /* finish CHOLMOD */
    return (0) ;
}
