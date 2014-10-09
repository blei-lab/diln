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

#ifndef GSL_WRAPPER_H_INCLUDED
#define GSL_WRAPPER_H_INCLUDED

void error(char * fmt);
// a function that terminates the program if anything is passed to
// the functions below that is of inconsistent dimensionality

void Ax(gsl_matrix * A, gsl_vector * x, gsl_vector * out, int trans);
// calculate the matrix-vector product out = A*x if trans = 0
// and out = A^T*x if trans = 1

double xTy(gsl_vector * x, gsl_vector * y);
// calculate the dot product dot = x*y

void AprodB(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C);
// calculate the matrix-matrix product C = A*B

void Gram(gsl_matrix * A, gsl_matrix * out);
// calculate the gram matrix A^T*A

void dotMult_vec(gsl_vector * x, gsl_vector * y, gsl_vector * out);
// calculate the element-wise product of vectors x and y

void dotMult_mat(gsl_matrix * A, gsl_matrix * B, gsl_matrix * C);
// calculate the element-wise product of matrices A and B

void sum(gsl_matrix * A, gsl_vector * out, int dim);
// sum the elements of matrix A along dimension dim

void x_sub_y(gsl_vector * x, gsl_vector * y, gsl_vector * out);
// subtract the vector y from x

void inv(gsl_matrix * X, gsl_matrix * invX);
// invert the matrix X

void vec_abs(gsl_vector * in, gsl_vector * out);
// return the absolute values of the elements in a vector

void normMat(gsl_matrix *X, gsl_matrix *Xnorm, int dim);
// normalize the vectors in X along dimension dim (L2 norm)

void normVec(gsl_vector *y, gsl_vector *out);
// normalize the vector y (L2 norm)

double vecGet(gsl_vector *x, int dim);
// return the value x[dim]

void vecSet(gsl_vector *x, int dim, double value);
// set x[dim] = value

double matGet(gsl_matrix *X, int d1, int d2);
// return the value X[d1][d2]

void matSet(gsl_matrix *X, int d1, int d2, double value);
// set X[d1][d2] = value

void vecScale(gsl_vector *y, gsl_vector *out, double a);
// scale a vector out = a*y

void colGet(gsl_matrix * X, gsl_vector * out, int col);
// return a column from the matrix X

double log_det(gsl_matrix *X);
// calculate and return ln|X|

void zeros(gsl_matrix *X, double value);
// zero out the values in X (and then replace with value)

void eye(gsl_matrix *X, double value);
// make X the identity matrix (and multiply by value)

double psi(double value);

double gammaln(double value);

#endif // GSL_WRAPPER_H_INCLUDED
