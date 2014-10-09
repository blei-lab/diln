// (C) Copyright 2010, John Paisley, Chong Wang and David M. Blei

// This file is part of DILN-C.

// DILN-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// DILN-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include "gsl_wrapper.h"

void error(char * fmt)
// a function that terminates the program if anything is passed to the functions below
// that is of inconsistent dimensionality
{
    printf(fmt);
    exit(1);
}

void Ax(gsl_matrix * A, gsl_vector * x, gsl_vector * out, int trans)
// calculate the matrix-vector product out = A*x if trans = 0
// and out = A^T*x if trans = 1
{
    if (trans == 0)
    {
        if (A->size2 != x->size)
            error("Error in Ax(): matrix*vector dimensions do not match");
        else if (out->size != A->size1)
            error("Error in Ax(): output vector is not correct size");

        gsl_vector * out2 = gsl_vector_alloc(out->size);
        int D = out->size;
        int N = A->size2;

        int n, d;
        double temp;
        for (d = 0; d < D; d++)
        {
            temp = 0;
            for (n = 0; n < N; n++)
            {
                temp = temp + gsl_matrix_get(A,d,n)*gsl_vector_get(x,n);
            }
            gsl_vector_set(out2,d,temp);
        }
        gsl_vector_memcpy(out,out2);
        gsl_vector_free(out2);
    }
    else if (trans == 1)
    {
        if (A->size1 != x->size)
            error("Error in Ax(): matrix*vector dimensions do not match");
        else if (out->size != A->size2)
            error("Error in Ax(): output vector is not correct size");

        gsl_vector * out2 = gsl_vector_alloc(out->size);
        int D = out->size;
        int M = A->size1;

        int m, d;
        double temp;
        for (d = 0; d < D; d++)
        {
            temp = 0;
            for (m = 0; m < M; m++)
            {
                temp = temp + gsl_matrix_get(A,m,d)*gsl_vector_get(x,m);
            }
            gsl_vector_set(out2,d,temp);
        }
        gsl_vector_memcpy(out,out2);
        gsl_vector_free(out2);
    }
    else
        error("Error in Ax(): trans must be 0 or 1");
}

double xTy(gsl_vector * x, gsl_vector * y)
// calculate the dot product dot = x*y
{
    if (x->size != y->size)
        error("Error in xTy(): vectors must have the same dimensionality");

    double dot = 0;
    int D = x->size;
    int d;

    for (d = 0; d < D; d++)
    {
        dot = dot + gsl_vector_get(x,d)*gsl_vector_get(y,d);
    }
    return dot;
}

void AprodB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C)
// calculate the matrix-matrix product C = A*B
{
    if (A->size2 != B->size1)
        error("Error in AprodB(): matrix*matrix dimensions do not match");
    else if (A->size1 != C->size1 || B->size2 != C->size2)
        error("Error in AprodB(): output matrix is not correct size");

    gsl_matrix * C2 = gsl_matrix_alloc(C->size1,C->size2);
    int M = A->size1;
    int N = B->size2;
    int K = A->size2;

    int m, n, k;
    double temp;
    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            temp = 0;
            for (k = 0; k < K; k++)
            {
                temp = temp + gsl_matrix_get(A,m,k)*gsl_matrix_get(B,k,n);
            }
            gsl_matrix_set(C2,m,n,temp);
        }
    }
    gsl_matrix_memcpy(C,C2);
    gsl_matrix_free(C2);
}

void Gram(gsl_matrix * A, gsl_matrix * out)
// calculate the gram matrix A^T*A
{
    int M = A->size1;
    int N = A->size2;

    if (out->size1 != N || out->size2 != N)
        error("Error in Gram(): output matrix dimensions incorrect");

    int m, n1, n2;
    double temp;
    for (n1 = 0; n1 < N; n1++)
    {
        for (n2 = 0; n2 < N; n2++)
        {
            temp = 0;
            for (m = 0; m < M; m++)
            {
                temp = temp + gsl_matrix_get(A,m,n1)*gsl_matrix_get(A,m,n2);
            }
            gsl_matrix_set(out,n1,n2,temp);
        }
    }
}

void dotMult_vec(gsl_vector * x, gsl_vector * y, gsl_vector * out)
// calculate the element-wise product of vectors x and y
{
    if (x->size != y->size)
        error("Error in dotMult_vec(): vectors must be the same size");

    int D = x->size;
    int d;
    double temp;
    for (d = 0; d < D; d++)
    {
        temp = gsl_vector_get(x,d)*gsl_vector_get(y,d);
        gsl_vector_set(out,d,temp);
    }
}

void dotMult_mat(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C)
// calculate the element-wise product of matrices A and B
{
    if (A->size1 != B->size1 || A->size2 != B->size2)
        error("Error in dotMult_mat(): matrix dimensions do not match");


    int M = A->size1;
    int N = A->size2;
    int m, n;
    double temp;
    for (m = 0; m < M; m++)
    {
        for (n = 0; n <N; n++)
        {
            temp = gsl_matrix_get(A,m,n)*gsl_matrix_get(B,m,n);
            gsl_matrix_set(C,m,n,temp);
        }
    }
}

void sum(gsl_matrix * A, gsl_vector * out, int dim)
// sum the elements of matrix A along dimension dim
{
    if (dim == 1)
    {
        if (out->size != A->size2)
            error("Error in sum(): input and output dimensions do not match");

        int N = A->size2;
        int M = A->size1;
        int n, m;
        double tmp;
        for (n = 0; n < N; n++)
        {
            tmp = 0;
            for (m = 0; m < M; m++)
            {
                tmp = tmp + gsl_matrix_get(A,m,n);
            }
            gsl_vector_set(out,n,tmp);
        }
    }
    else if (dim == 2)
    {
        if (out->size != A->size1)
            error("Error in sum(): input and output dimensions do not match");

        int N = A->size2;
        int M = A->size1;
        int n, m;
        double tmp;
        for (m = 0; m < M; m++)
        {
            tmp = 0;
            for (n = 0; n < N; n++)
            {
                tmp = tmp + gsl_matrix_get(A,m,n);
            }
            gsl_vector_set(out,m,tmp);
        }
    }
    else
    {
        error("Error in sum(): dim must take the value 1 or 2");
    }
}

void x_sub_y(gsl_vector * x, gsl_vector * y, gsl_vector * out)
// subtract the vector y from x
{
    if (x->size != y->size)
        error("Error in x_sub_y(): vectors must be the same dimension");

    int D = x->size;
    int d;
    double temp;
    for (d = 0; d < D; d++)
    {
        temp = gsl_vector_get(x,d) - gsl_vector_get(y,d);
        gsl_vector_set(out,d,temp);
    }
}

void inv(gsl_matrix * X, gsl_matrix * invX)
// invert the matrix X
{
    int signum;
    gsl_matrix * LU = gsl_matrix_calloc(X->size1,X->size2);
    gsl_permutation * p = gsl_permutation_calloc(X->size1);
    gsl_matrix_memcpy(LU,X);

    gsl_linalg_LU_decomp (LU,p,&signum);
    gsl_linalg_LU_invert(LU,p,invX);

    gsl_matrix_free(LU);
    gsl_permutation_free(p);
}

void vec_abs(gsl_vector * in, gsl_vector * out)
// return the absolute values of the elements in a vector
{
    int D = in->size;
    int d;
    for (d = 0; d < D; d++)
    {
        if (gsl_vector_get(in,d) < 0)
            gsl_vector_set(out,d,-1*gsl_vector_get(in,d));
        else
            gsl_vector_set(out,d,gsl_vector_get(in,d));
    }
}

void normMat(gsl_matrix *X, gsl_matrix *Xnorm, int dim)
// normalize the vectors in X along dimension dim (L2 norm)
{
    if (X->size1 != Xnorm->size1 || X->size2 != Xnorm->size2)
        error("Error in normMat(): dimensions of output matrix must equal input matrix");

    int M = X->size1;
    int N = X->size2;
    int n, m;
    double temp;

    if (dim == 1)
    {
        for (n = 0; n < N; n++)
        {
            temp = 0;
            for (m = 0; m < M; m++)
            {
                temp = temp + pow(gsl_matrix_get(X,m,n),2);
            }
            temp = sqrt(temp);
            for (m = 0; m < M; m++)
            {
                gsl_matrix_set(Xnorm,m,n,gsl_matrix_get(X,m,n)/temp);
            }
        }
    }
    else if (dim == 2)
    {
        for (m = 0; m < M; m++)
        {
            temp = 0;
            for (n = 0; n < N; n++)
            {
                temp = temp + pow(gsl_matrix_get(X,m,n),2);
            }
            temp = sqrt(temp);
            for (n = 0; n < N; n++)
            {
                gsl_matrix_set(Xnorm,m,n,gsl_matrix_get(X,m,n)/temp);
            }
        }
    }
    else
        error("Error in normMat(): dim takes the value 1 or 2");
}

void normVec(gsl_vector *y, gsl_vector *out)
// normalize the vector y (L2 norm)
{
    int K = y->size;
    double val = 0;
    int k;
    for (k = 0; k < K; k++)
    {
        val = val + pow(vecGet(y,k),2);
    }
    val = sqrt(val);
    for (k = 0; k < K; k++)
    {
        vecSet(out,k,vecGet(y,k)/val);
    }
}

double vecGet(gsl_vector *x, int dim)
// return the value x[dim]
{
    double value = gsl_vector_get(x,dim);
    return value;
}

void vecSet(gsl_vector *x, int dim, double value)
// set x[dim] = value
{
    gsl_vector_set(x,dim,value);
}

double matGet(gsl_matrix *X, int d1, int d2)
// return the value X[d1][d2]
{
    double value = gsl_matrix_get(X,d1,d2);
    return value;
}

void matSet(gsl_matrix *X, int d1, int d2, double value)
// set X[d1][d2] = value
{
    gsl_matrix_set(X,d1,d2,value);
}

void vecScale(gsl_vector *y, gsl_vector *out, double a)
// scale a vector out = a*y
{
    int K = y->size;
    int k;
    for (k = 0; k < K; k++)
    {
        vecSet(out,k,a*vecGet(y,k));
    }
}

void colGet(gsl_matrix * X, gsl_vector * out, int col)
// return a column from the matrix X
{
    int K = out->size;
    int k;
    for (k = 0; k < K; k++)
    {
        vecSet(out,k,matGet(X,k,col));
    }
}

double log_det(gsl_matrix *X)
// calculate and return ln|X|
{
    if (X->size1 != X->size2)
        error("Error in log_det(): matrix must be square");

    int i, j;
    int N = X->size1;
    for (i = 0; i < N; i++)
    {
        for (j = i; j < N; j++)
        {
            if (matGet(X,i,j) != matGet(X,j,i))
                error("Error in log_det(): matrix must be symmetric");
        }
    }

    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(N);
    gsl_vector * eval = gsl_vector_alloc(N);
    gsl_eigen_symm(X,eval,w);
    double detVal = 0;
    double temp;
    for (i = 0; i < N; i++)
    {
        temp = vecGet(eval,i);
        if (temp == 0)
            error("Error in log_det(): matrix is rank deficient");

        detVal = detVal + log(temp);
    }
    gsl_eigen_symm_free(w);

    return detVal;
}

void zeros(gsl_matrix *X, double value)
// zero out the values in X
{
    int M = X->size1;
    int N = X->size2;
    int m, n;
    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            matSet(X,m,n,value);
        }
    }
}

void eye(gsl_matrix *X, double value)
// make X the identity matrix
{
    int N = X->size1;
    int n1, n2;
    if (X->size2 != N)
        error("Error in eye: matrix is not square");

    for (n1 = 0; n1 < N; n1++)
    {
        for (n2 = 0; n2 < N; n2++)
        {
            if (n1 == n2)
                matSet(X,n1,n2,value);
            else
                matSet(X,n1,n2,0);
        }
    }
}

double psi(double value)
{
    return gsl_sf_psi(value);
}

double gammaln(double value)
{
    return gsl_sf_lngamma(value);
}
