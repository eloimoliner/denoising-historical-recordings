/*=================================================================
 *
 * toeplitzC.C	Sample .MEX file corresponding to toeplitz.m
 *	        Solves simple 3 body orbit problem 
 *
 * The calling syntax is:
 *
 *		[yp] = yprime(t, y)
 *   TOEPLITZ(C,R) is a non-symmetric Toeplitz matrix having C as its
 *   first column and R as its first row.   
 *
 *   TOEPLITZ(R) is a symmetric Toeplitz matrix for real R.
 *   For a complex vector R with a real first element, T = toeplitz(r) 
 *   returns the Hermitian Toeplitz matrix formed from R. When the 
 *   first element of R is not real, the resulting matrix is Hermitian 
 *   off the main diagonal, i.e., T_{i,j} = conj(T_{j,i}) for i ~= j.
 *
 *  You may also want to look at the corresponding M-code, yprime.m.
 *
 * This is a MEX-file for MATLAB.  
 * Copyright 1984-2006 The MathWorks, Inc.
 *
 *=================================================================*/
/* $Revision: 1.10.6.4 $ */
#include <math.h>
#include "mex.h"

/* Input Arguments */

#define	C_IN	prhs[0]
#define	R_IN	prhs[1]


/* Output Arguments */

#define	T_OUT	plhs[0]


static void toeplitzC(
		   double	t[],
		   double	c[],
 		   double	r[],
           int m, int n
		   )
{
    int i,j,m0;

    for (j=0;j<n;j++){
        m0 = j>m?m:j;
        for (i=0;i<=m0;i++)
            t[j*m+i] = r[j-i];
        for (i=j+1;i<m;i++)
            t[j*m+i] = c[i-j];
    }
    return;
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *tr,*ti; 
    double *cr,*ci,*rr,*ri; 
    mwSize tm,tn; 
    
    /* Check for proper number of arguments */
    
    if (nrhs > 2) { 
	mexErrMsgTxt("More than 2 input arguments."); 
    } else if (nrhs == 0) {
    mexErrMsgTxt("1 or 2 input arguments required."); 
    }  else if (nrhs == 1) {
    mexErrMsgTxt("1 input argument: not implemented (yet), use toeplitz.");
    } else if (nlhs > 1) {
	mexErrMsgTxt("Too many output arguments."); 
    } 
    
    if (nrhs == 1) { 
	mexErrMsgTxt("Not implemented (yet). Please use 2 input arguments or toeplitz.m with one input argument."); 
    }
    
    /* Check the dimensions of Y.  Y can be 4 X 1 or 1 X 4. */ 
    
    tm = mxGetNumberOfElements(C_IN);
    tn = mxGetNumberOfElements(R_IN);
    
    /* Create a matrix for the return argument */
    if (mxIsComplex(C_IN) || mxIsComplex(R_IN))
        T_OUT = mxCreateDoubleMatrix(tm, tn, mxCOMPLEX);
    else
        T_OUT = mxCreateDoubleMatrix(tm, tn, mxREAL);
    
    /* Assign pointers to the various parameters */ 
    tr = mxGetPr(T_OUT);
    
    cr = mxGetPr(C_IN); 
    rr = mxGetPr(R_IN);
        
    /* Do the actual computations in a subroutine */
    toeplitzC(tr,cr,rr,tm,tn); 
    
    /* Imaginary part */
    if (mxIsComplex(C_IN) || mxIsComplex(R_IN)){
        /*if (!mxIsComplex(C_IN)){
            mexErrMsgTxt("Not implemented (yet). A");
        }
        else*/
        if (mxIsComplex(C_IN))
            ci = mxGetPi(C_IN);
        else
            ci = mxGetPr(mxCreateDoubleMatrix(tm, 1, mxREAL));
        /*if (!mxIsComplex(R_IN)){
            mexErrMsgTxt("Not implemented (yet). B");
        }
        else*/
        if (mxIsComplex(R_IN))
            ri = mxGetPi(R_IN);
        else
            ri = mxGetPr(mxCreateDoubleMatrix(tn, 1, mxREAL));
        ti = mxGetPi(T_OUT);
        toeplitzC(ti, ci, ri, tm, tn);
    }

    return;
    
}


