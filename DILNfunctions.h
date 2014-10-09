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

#ifndef DILNFUNCTIONS_H_INCLUDED
#define DILNFUNCTIONS_H_INCLUDED
#include "importData.h"

double multEntropy(doc thisdoc, gsl_matrix * C);

double DirEntropy(gsl_matrix * Gam, gsl_matrix * ElnGam);

double gamEntropy(gsl_matrix * A, gsl_matrix * B);

double LboundDataLevel(doc thisdoc, gsl_matrix * C, gsl_matrix * ElnGam, gsl_matrix * ElnZ, int m);

double up_alpha(int K, gsl_vector * V);

double get_step_beta(double beta, double dbeta, gsl_vector * p, gsl_vector * sum_mu, gsl_vector * sumElnZ, int M);

double up_beta(double beta, gsl_vector * p, gsl_matrix * mu, gsl_matrix * A, gsl_matrix * B);

void V_init(gsl_vector * V);

void stick_break(gsl_vector * p, gsl_vector * V);

// ***** VB-E Step *****
double VB_Estep(corpus * allDocs, gsl_matrix * Gam, gsl_matrix * A, gsl_matrix * B, gsl_matrix * N, double gamma);

// ***** Update A and B *****
double VB_Mstep_AB(corpus * allDocs,gsl_matrix * A, gsl_matrix * B,gsl_matrix * N, double beta,gsl_vector * p, gsl_matrix * mu, gsl_matrix * sig);

// ***** Update V1,...,VK *****
void stick_break_left(gsl_vector * V, gsl_vector * out);

double get_step_V(gsl_vector * V, gsl_vector * dV, double alpha, double beta, gsl_vector * sum_mu, gsl_vector * sum_psiA_minlnB, int M);

double VB_Mstep_V(gsl_matrix * A, gsl_matrix * B, gsl_matrix * mu, gsl_vector * V, gsl_vector * p, double alpha, double beta);

// ***** Update mu and sigma *****
double getLboundMuVuKern(gsl_vector * u, gsl_matrix * invKern, gsl_matrix * mu, gsl_matrix * sig);

double get_objective_MuV(gsl_vector * mu_check, gsl_vector * sig_check, gsl_vector * p, gsl_vector * AdivB, gsl_vector * u, gsl_matrix * invKern, double beta, int m);

double get_step_MuV(gsl_vector * mu, gsl_vector * sig, gsl_vector * dmu, gsl_vector * dsig, gsl_vector * p, gsl_vector * AdivB, gsl_vector * u, gsl_matrix * invKern, double beta, int m);

double VB_Mstep_lognorm(gsl_matrix * mu, gsl_matrix * sig, gsl_matrix * A, gsl_matrix * B, gsl_vector * p, gsl_vector * u, gsl_matrix * Kern, double beta);

// ***** Kmeans Initialization *****
void centInit(gsl_matrix * X, gsl_matrix * C, int K, int D, int N);

void indUpdate(gsl_matrix *X, gsl_matrix *C, int indicator[], int K, int D, int N, int pnorm);

void centUpdate(gsl_matrix *X, gsl_matrix *C, int count[], int indicator[], int K, int D, int N);

void Kmeans_init(corpus * allDocs, gsl_matrix * Gam, int pnorm, double gamma, int numIte);

#endif // DILNFUNCTIONS_H_INCLUDED
