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
#include <gsl/gsl_rng.h>
#include "gsl_wrapper.h"
#include "DILNfunctions.h"
#include "importData.h"

double multEntropy(doc thisdoc, gsl_matrix * C)
{
    double val = 0;
    double temp = 0;
    double eps = pow(10,-16);
    int k, n;
    int wordCnt;
    for (n = 0; n < C->size2; n++)
    {
        for (k = 0; k < C->size1; k++)
        {
            wordCnt = vecGet(thisdoc.wordCnt,n);
            temp = matGet(C,k,n);
            val = val - temp*log(temp+eps)*wordCnt;
        }
    }
    return val;
}

double DirEntropy(gsl_matrix * Gam, gsl_matrix * ElnGam)
{
    double val = 0;
    double thisval;
    int K = Gam->size1;
    int D = Gam->size2;
    int k, d;
    gsl_vector * sumGam = gsl_vector_alloc(K);
    sum(Gam,sumGam,2);
    for (k = 0; k < K; k++)
    {
        val = val - gammaln(vecGet(sumGam,k));
        for (d = 0; d < D; d++)
        {
            thisval = matGet(Gam,k,d);
            val = val + gammaln(thisval);
            val = val - (thisval-1)*matGet(ElnGam,k,d);
        }
    }
    gsl_vector_free(sumGam);
    return val;
}

double gamEntropy(gsl_matrix * A, gsl_matrix * B)
{
    double val = 0;
    int K = A->size1;
    int M = A->size2;
    int k, m;
    double a;
    for (m = 0; m < M; m++)
    {
        for (k = 0; k < K; k++)
        {
            a = matGet(A,k,m);
            val = val + gammaln(a) - (a-1)*psi(a) - log(matGet(B,k,m)) + a;
        }
    }
    return val;
}

double LboundDataLevel(doc thisdoc, gsl_matrix * C, gsl_matrix * ElnGam, gsl_matrix * ElnZ, int m)
{
    double val = 0;
    int K = C->size1;
    int N = C->size2;
    int k, n;
    double prob;
    int wordCnt, wordIdx;
    for (n = 0; n < N; n++)
    {
        wordCnt = vecGet(thisdoc.wordCnt,n);
        wordIdx = vecGet(thisdoc.wordIdx,n);
        for (k = 0; k < K; k++)
        {
            prob = matGet(C,k,n);
            val = val + wordCnt*prob*matGet(ElnGam,k,wordIdx);
            val = val + wordCnt*prob*matGet(ElnZ,k,m);
        }
    }
    return val;
}

void V_init(gsl_vector * V)
{
    double K = V->size;
    int k;
    double prod = 1;
    for (k = 0; k < K; k++)
    {
        vecSet(V,k,(1/K)/prod);
        prod = prod*(1-vecGet(V,k));
    }
}

void stick_break(gsl_vector * p, gsl_vector * V)
// this function assumes V[K-1] = 1
{
    double K = V->size;
    int k;
    double prod = 1;
    for (k = 0; k < K; k++)
    {
        vecSet(p,k,vecGet(V,k)*prod);
        prod = prod*(1-vecGet(V,k));
    }
}

double VB_Estep(corpus * allDocs, gsl_matrix * Gam, gsl_matrix * A, gsl_matrix * B, gsl_matrix * N, double gamma)
// perform the E-step for the DILN model
{
    double Lbound = 0;
    int K = A->size1;
    int M = A->size2;
    int D = Gam->size2;
    int k, m, w, d;
    double val;
    gsl_matrix * ElnGam = gsl_matrix_alloc(K,Gam->size2);
    gsl_matrix * ElnZ = gsl_matrix_alloc(K,M);
    gsl_vector * sumGam = gsl_vector_alloc(K);
    sum(Gam,sumGam,2);
    for (k = 0; k < K; k++)
    {
        for (m = 0; m < M; m++)
        {
            val = psi(matGet(A,k,m)) - log(matGet(B,k,m));
            matSet(ElnZ,k,m,val);
        }
        for (d = 0; d < D; d++)
        {
            val = psi(matGet(Gam,k,d)) - psi(vecGet(sumGam,k));
            matSet(ElnGam,k,d,val);
            Lbound = Lbound + (gamma/D - 1)*val;
        }
    }
    Lbound = Lbound + DirEntropy(Gam,ElnGam);
    zeros(Gam,gamma/D);
    zeros(N,0);

//  calculate latent topic probabilities on the word level
    int numUnique;
    int wordIdx;
    int wordCnt;
    for (m = 0; m < M; m++)
    {
        numUnique = allDocs->docs[m].numUnique;
        gsl_vector * maxloglik = gsl_vector_alloc(numUnique);
        gsl_matrix * C = gsl_matrix_alloc(K,numUnique);
        for (w = 0; w < numUnique; w++)
        {
            vecSet(maxloglik,w,-INFINITY);
            wordIdx = vecGet(allDocs->docs[m].wordIdx,w);
            for (k = 0; k < K; k++)
            {
                val = matGet(ElnGam,k,wordIdx) + matGet(ElnZ,k,m);
                matSet(C,k,w,val);
                if (val > vecGet(maxloglik,w))
                    vecSet(maxloglik,w,val);
            }
        }
        for (w = 0; w < numUnique; w++)
        {
            val = 0;
            for (k = 0; k < K; k++)
            {
                matSet(C,k,w,exp(matGet(C,k,w)-vecGet(maxloglik,w)));
                val = val + matGet(C,k,w);
            }
            for (k = 0; k < K; k++)
                matSet(C,k,w,matGet(C,k,w)/val);

        //  Update Gamma matrix and counts matrix
            wordCnt = vecGet(allDocs->docs[m].wordCnt,w);
            wordIdx = vecGet(allDocs->docs[m].wordIdx,w);
            for (k = 0; k < K; k++)
            {
                val = matGet(C,k,w)*wordCnt;
                matSet(Gam,k,wordIdx,matGet(Gam,k,wordIdx)+val);
                matSet(N,k,m,matGet(N,k,m)+val);
            }
        }
    //  Calculate lower bound update
        Lbound = Lbound + multEntropy(allDocs->docs[m],C);
        Lbound = Lbound + LboundDataLevel(allDocs->docs[m],C,ElnGam,ElnZ,m);
        gsl_vector_free(maxloglik);
        gsl_matrix_free(C);
    }
    gsl_matrix_free(ElnGam);
    gsl_matrix_free(ElnZ);
    gsl_vector_free(sumGam);
    return Lbound;
}

double up_alpha(int K, gsl_vector * V)
{
    double alpha;
    double val = 0;
    int k;
    for (k = 0; k < K-1; k++)
    {
        val = val + log(1-vecGet(V,k));
    }
    alpha = -(K-1)/val;
    return alpha;
}

double get_step_beta(double beta, double dbeta, gsl_vector * p, gsl_vector * sum_mu, gsl_vector * sumElnZ, int M)
{
    double rho = .8;
    double stepsize = 1/rho;
    double f, fold = -INFINITY;
    double beta_check;
    int bool_stop = 0, K = p->size, k;
    while (bool_stop == 0)
    {
        stepsize = rho*stepsize;
        beta_check = beta + stepsize*dbeta;
        f = 0;
        for (k = 0; k < K; k++)
        {
            f = beta_check*vecGet(p,k)*(vecGet(sum_mu,k) + vecGet(sumElnZ,k)) - M*gammaln(beta_check*vecGet(p,k));
        }
        if (f > fold) fold = f;
        else bool_stop = 1;
    }
    stepsize = stepsize/rho;
    return stepsize;
}

double up_beta(double beta, gsl_vector * p, gsl_matrix * mu, gsl_matrix * A, gsl_matrix * B)
{
    double dbeta, temp, stepsize;
    int K = mu->size1;
    int M = mu->size2;
    gsl_vector * sum_mu = gsl_vector_alloc(K);
    gsl_vector * sumElnZ = gsl_vector_calloc(K);
    sum(mu,sum_mu,2);
    int k, m, s;
    for (k = 0; k < K; k++)
    {
        temp = 0;
        for (m = 0; m < M; m++)
        {
            temp = temp + psi(matGet(A,k,m)) - log(matGet(B,k,m));
        }
        vecSet(sumElnZ,k,temp);
    }
    int numstep = 5; // this effectively bounds the maximum absolute change for an iteration
    for (s = 0; s < numstep; s++)
    {
        dbeta = 0;
        for (k = 0; k < K; k++)
        {
            dbeta = dbeta + vecGet(p,k)*(vecGet(sum_mu,k) - M*psi(beta*vecGet(p,k)) + vecGet(sumElnZ,k));
        }
        if (dbeta > 1) dbeta = 1;
        else if (dbeta < -1) dbeta = -1;
        if (dbeta + beta < 0) dbeta = -.9*beta;
        stepsize = get_step_beta(beta,dbeta,p,sum_mu,sumElnZ,M);
        beta = beta + stepsize*dbeta;
    }
    gsl_vector_free(sum_mu);
    gsl_vector_free(sumElnZ);
    return beta;
}

double VB_Mstep_AB(corpus * allDocs, gsl_matrix * A, gsl_matrix * B,gsl_matrix * N, double beta, gsl_vector * p, gsl_matrix * mu, gsl_matrix * sig)
// update A and B
{
    double Lbound = 0;
    int K = A->size1;
    int M = A->size2;
    int k, m;
    gsl_matrix * expMUV = gsl_matrix_alloc(K,M);
    gsl_matrix * AdivB = gsl_matrix_alloc(K,M);
    gsl_vector * Taylor_param = gsl_vector_calloc(M);
    for (k = 0; k < K; k++)
    {
        for (m = 0; m < M; m++)
        {
            matSet(expMUV,k,m,exp(matGet(mu,k,m)+.5*matGet(sig,k,m)));
            matSet(AdivB,k,m,matGet(A,k,m)/matGet(B,k,m));
        }
    }
    sum(AdivB,Taylor_param,1);

    double val1, val2;
    for (m = 0; m < M; m++)
    {
        Lbound = Lbound - allDocs->docs[m].numWords*log(vecGet(Taylor_param,m));
        for (k = 0; k < K; k++)
        {
            val1 = matGet(N,k,m) + beta*vecGet(p,k);
            matSet(A,k,m,val1);
            val2 = allDocs->docs[m].numWords/vecGet(Taylor_param,m) + matGet(expMUV,k,m);
            matSet(B,k,m,val2);
            Lbound = Lbound - matGet(AdivB,k,m)*matGet(expMUV,k,m);
        }
    }

    gsl_vector * Taylor_param2 = gsl_vector_calloc(M);
    for (k = 0; k < K; k++)
    {
        for (m = 0; m < M; m++)
        {
            matSet(AdivB,k,m,matGet(A,k,m)/matGet(B,k,m));
        }
    }
    sum(AdivB,Taylor_param2,1);
    for (m = 0; m < M; m++)
        Lbound = Lbound - allDocs->docs[m].numWords*(vecGet(Taylor_param2,m) - vecGet(Taylor_param,m))/vecGet(Taylor_param,m);

    gsl_matrix_free(expMUV);
    gsl_matrix_free(AdivB);
    gsl_vector_free(Taylor_param);
    gsl_vector_free(Taylor_param2);

    Lbound = Lbound + gamEntropy(A,B);
    return Lbound;
}

// ****************** FUNCTIONS FOR UPDATING V_k *********************

void stick_break_left(gsl_vector * V, gsl_vector * out)
{
    int K = V->size;
    int k;
    double val;
    vecSet(out,0,1);
    for (k = 1; k < K; k++)
    {
        val = vecGet(out,k-1)*(1-vecGet(V,k));
        vecSet(out,k,val);
    }
}

double get_step_V(gsl_vector * V, gsl_vector * dV, double alpha, double beta, gsl_vector * sum_mu, gsl_vector * sum_psiA_minlnB, int M)
// learn the step size for V_1,...,V_K
{
    double stepsize, t1, t2;
    int K = V->size;
    int k;
    double maxstep = 10000000;
    for (k = 0; k < K-1; k++)
    {
        t1 = -vecGet(V,k)/vecGet(dV,k);
        t2 = (1-vecGet(V,k))/vecGet(dV,k);
        if (t1 > 0 && t1 < maxstep)
            maxstep = t1;
        else if (t2 > 0 && t2 < maxstep)
            maxstep = t2;
    }
    maxstep = .95*maxstep;
    stepsize = maxstep;

    gsl_vector * Vcheck = gsl_vector_alloc(K);
    gsl_vector * pcheck = gsl_vector_alloc(K);
    int bool_stop = 0;
    double rho = .75;
    double f = 0, fold = -INFINITY;
    while (bool_stop == 0)
    {
        for (k = 0; k < K-1; k++)
        {
            vecSet(Vcheck,k,vecGet(V,k)+stepsize*vecGet(dV,k));
        }
        vecSet(Vcheck,K-1,1);
        stick_break(pcheck,Vcheck);
        f = 0;
        for (k = 0; k < K; k++)
        {
            f = f + beta*vecGet(sum_mu,k)*vecGet(pcheck,k) - M*gammaln(beta*vecGet(pcheck,k));
            f = f + beta*vecGet(pcheck,k)*vecGet(sum_psiA_minlnB,k);
            if (k < K-1) f = f + (alpha-1)*log(1-vecGet(Vcheck,k));
        }
        if (f > fold)
        {
            fold = f;
            stepsize = stepsize*rho;
        }
        else
        {
            bool_stop = 1;
            stepsize = stepsize/rho;
        }
    }
    gsl_vector_free(pcheck);
    gsl_vector_free(Vcheck);
    return stepsize;
}

double VB_Mstep_V(gsl_matrix * A, gsl_matrix * B, gsl_matrix * mu, gsl_vector * V, gsl_vector * p, double alpha, double beta)
// update V1,...,VK using gradient ascent
{
    int K = A->size1;
    int M = A->size2;
    int k, m, s, l;
    int numStep = 20;
    double steplength, val, val2;
    gsl_vector * dV = gsl_vector_calloc(K);
    gsl_matrix * psiA = gsl_matrix_calloc(K,M);
    gsl_matrix * lnB = gsl_matrix_calloc(K,M);
    gsl_vector * sum_psiA_minlnB = gsl_vector_calloc(K);
    gsl_vector * sum_mu = gsl_vector_calloc(K);
    gsl_vector * stick_left = gsl_vector_calloc(K);
    gsl_vector * psi_stick = gsl_vector_calloc(K);
    sum(mu,sum_mu,2);
    stick_break_left(V,stick_left);
    for (k = 0; k < K; k++)
    {
        for (m = 0; m < M; m++)
        {
            matSet(psiA,k,m,psi(matGet(A,k,m)));
            matSet(lnB,k,m,log(matGet(B,k,m)));
        }
    }
    gsl_matrix_sub(psiA,lnB);
    sum(psiA,sum_psiA_minlnB,2);
    for (s = 0; s < numStep; s++)
    {
        for (k = 0; k < K; k++)
        {
            vecSet(psi_stick,k,psi(beta*vecGet(p,k)));
        }
        for (k = 0; k < K-1; k++)
        {
            val = -(alpha-1)/(1-vecGet(V,k));
            val = val + beta*vecGet(stick_left,k)*(vecGet(sum_mu,k) - M*vecGet(psi_stick,k) + vecGet(sum_psiA_minlnB,k));
            val2 = 0;
            for (l = k+1; l < K; l++)
            {
                val2 = val2 + (M*vecGet(psi_stick,l) - vecGet(sum_mu,l) - vecGet(sum_psiA_minlnB,l))*vecGet(p,l)/(1-vecGet(V,k));
            }
            val = val + beta*val2;
            vecSet(dV,k,val);
        }
        vecSet(dV,K-1,0);
        normVec(dV,dV);
        steplength = get_step_V(V,dV,alpha,beta,sum_mu,sum_psiA_minlnB,M);
        for (k = 0; k < K; k++)
        {
            vecSet(V,k,vecGet(V,k)+steplength*vecGet(dV,k));
        }
        vecSet(V,K-1,1);
        stick_break(p,V);
    }
    gsl_vector_free(stick_left);
    gsl_vector_free(sum_psiA_minlnB);
    gsl_vector_free(psi_stick);
    gsl_vector_free(sum_mu);
    gsl_vector_free(dV);
    gsl_matrix_free(psiA);
    gsl_matrix_free(lnB);

    double Lbound = 0;
    for (k = 0; k < K; k++)
    {
        Lbound = Lbound + beta*vecGet(sum_mu,k)*vecGet(p,k) - M*gammaln(beta*vecGet(p,k));
        Lbound = Lbound + (beta*vecGet(p,k)-1)*vecGet(sum_psiA_minlnB,k);
        if (k < K-1) Lbound = Lbound + (alpha-1)*log(1-vecGet(V,k));
    }
    return Lbound;
}

// ******************** FUNCTIONS FOR UPDATING MU AND SIGMA *********************

double getLboundMuVuKern(gsl_vector * u, gsl_matrix * invKern, gsl_matrix * mu, gsl_matrix * sig)
{
    double Lbound = 0;
    int K = u->size;
    int M = mu->size2;
    gsl_vector * eigs = gsl_vector_alloc(K);
    gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(K);
    gsl_eigen_symm(invKern,eigs,w);
    int k, m;
    for (k = 0; k < K; k++)
    {
        Lbound = Lbound - .5*M*log(vecGet(eigs,k));
        for (m = 0; m < M; m++)
        {
            Lbound = Lbound + .5*log(matGet(sig,k,m));
            Lbound = Lbound - .5*matGet(sig,k,m)*matGet(invKern,k,k);
        }
    }
    gsl_vector * temp = gsl_vector_alloc(K);
    gsl_vector * colmu = gsl_vector_alloc(K);
    for (m = 0; m < M; m++)
    {
        colGet(mu,colmu,m);
        x_sub_y(colmu,u,colmu);
        Ax(invKern,colmu,temp,0);
        for (k = 0; k < K; k++)
        {
            Lbound = Lbound - .5*vecGet(temp,k)*vecGet(colmu,k);
        }
    }
    gsl_vector_free(temp);
    gsl_vector_free(colmu);
    gsl_eigen_symm_free(w);
    gsl_vector_free(eigs);
    return Lbound;
}

double get_objective_MuV(gsl_vector * mu_check, gsl_vector * sig_check, gsl_vector * p, gsl_vector * AdivB, gsl_vector * u, gsl_matrix * invKern, double beta, int m)
{
    double val = 0;
    int k;
    int K = mu_check->size;
    gsl_vector * temp = gsl_vector_alloc(K);
    gsl_vector * temp2 = gsl_vector_alloc(K);
    x_sub_y(mu_check,u,temp);
    Ax(invKern,temp,temp2,0);
    for (k = 0; k < K; k++)
    {
        val = val - vecGet(AdivB,k)*exp(vecGet(mu_check,k)+.5*exp(vecGet(sig_check,k))) + beta*vecGet(mu_check,k)*vecGet(p,k);
        val = val + .5*vecGet(sig_check,k) - .5*matGet(invKern,k,k)*exp(vecGet(sig_check,k));
        val = val - .5*vecGet(temp,k)*vecGet(temp2,k);
    }
    gsl_vector_free(temp);
    gsl_vector_free(temp2);
    return val;
}

double get_step_MuV(gsl_vector * mu, gsl_vector * sig, gsl_vector * dmu, gsl_vector * dsig, gsl_vector * p, gsl_vector * AdivB, gsl_vector * u, gsl_matrix * invKern, double beta, int m)
{
    double stepsize, rho;
    double rho_up = 1.25;
    double rho_dn = .75;
    double f_dn, f, f_up, f_old;
    int K = mu->size;
    int k;
    gsl_vector * mu_check = gsl_vector_alloc(K);
    gsl_vector * sig_check = gsl_vector_alloc(K);

    gsl_vector_memcpy(mu_check,mu);
    gsl_vector_memcpy(sig_check,sig);
    f = get_objective_MuV(mu_check,sig_check,p,AdivB,u,invKern,beta,m);

    for (k = 0; k < K; k++)
    {
        vecSet(mu_check,k,vecGet(mu,k)+rho_up*vecGet(dmu,k));
        vecSet(sig_check,k,vecGet(sig,k)+rho_up*vecGet(dsig,k));
    }
    f_up = get_objective_MuV(mu_check,sig_check,p,AdivB,u,invKern,beta,m);

    for (k = 0; k < K; k++)
    {
        vecSet(mu_check,k,vecGet(mu,k)+rho_dn*vecGet(dmu,k));
        vecSet(sig_check,k,vecGet(sig,k)+rho_dn*vecGet(dsig,k));
    }
    f_dn = get_objective_MuV(mu_check,sig_check,p,AdivB,u,invKern,beta,m);

    int bool_step = 1;
    if (f_up > f)
    {
        f_old = f_up;
        rho = rho_up;
        stepsize = rho;
    }
    else if (f_dn > f)
    {
        f_old = f_dn;
        rho = rho_dn;
        stepsize = rho;
    }
    else
    {
        bool_step = 0;
        stepsize = 1;
    }

    while (bool_step == 1)
    {
        stepsize = stepsize*rho;
        for (k = 0; k < K; k++)
        {
            vecSet(mu_check,k,vecGet(mu,k)+rho*vecGet(dmu,k));
            vecSet(sig_check,k,vecGet(sig,k)+rho*vecGet(dsig,k));
        }
        f = get_objective_MuV(mu_check,sig_check,p,AdivB,u,invKern,beta,m);
        if (f > f_old)
        {
            f_old = f;
        }
        else
        {
            bool_step = 0;
            stepsize = stepsize/rho;
        }
    }
    gsl_vector_free(mu_check);
    gsl_vector_free(sig_check);
    return stepsize;
}

double VB_Mstep_lognorm(gsl_matrix * mu, gsl_matrix * sig, gsl_matrix * A, gsl_matrix * B, gsl_vector * p, gsl_vector * u, gsl_matrix * Kern, double beta)
// update mu and sigma using gradient ascent
{
    int K = A->size1;
    int M = A->size2;
    gsl_vector * AdivB = gsl_vector_alloc(K);
    gsl_vector * expMuV = gsl_vector_alloc(K);
    gsl_matrix * invKern = gsl_matrix_alloc(K,K);
    gsl_vector * dmu = gsl_vector_calloc(K);
    gsl_vector * dsig = gsl_vector_calloc(K);
    gsl_vector * invKmu = gsl_vector_alloc(K);
    gsl_vector * colmu = gsl_vector_alloc(K);
    gsl_vector * colsig = gsl_vector_alloc(K);
    inv(Kern,invKern);
    double Lbound = getLboundMuVuKern(u,invKern,mu,sig);  // do this first to reduce duplicate computations

//  update mu and sig *** using the log-barrier method for sig ***
    int numstep = 1;
    int k, m, s;
    double stepsize, temp;
    for (m = 0; m < M; m++)
    {
        for (k = 0; k < K; k++)
        {
            vecSet(AdivB,k,matGet(A,k,m)/matGet(B,k,m));
        }
        for (s = 0; s < numstep; s++)
        {
            colGet(sig,colsig,m);
            colGet(mu,colmu,m);
            Ax(invKern,colmu,invKmu,0);
            for (k = 0; k < K; k++)
            {
                vecSet(expMuV,k,exp(matGet(mu,k,m)+.5*matGet(sig,k,m)));
                vecSet(dmu,k,beta*vecGet(p,k)-vecGet(AdivB,k)*vecGet(expMuV,k)-vecGet(invKmu,k));
//                vecSet(dsig,k,-.5*vecGet(AdivB,k)*vecGet(expMuV,k)-.5*matGet(invKern,k,k)+.5/matGet(sig,k,m));
                vecSet(dsig,k,-.5*vecGet(AdivB,k)*vecGet(expMuV,k)*matGet(sig,k,m)-.5*matGet(invKern,k,k)*matGet(sig,k,m)+.5);
            }
            temp = sqrt(xTy(dmu,dmu));
            for (k = 0; k < K; k++)
            {
                vecSet(dmu,k,vecGet(dmu,k)/temp);
                vecSet(dsig,k,vecGet(dsig,k)/temp);
            }
            stepsize = get_step_MuV(colmu,colsig,dmu,dsig,p,AdivB,u,invKern,beta,m);
            for (k = 0; k < K; k++)
            {
                matSet(mu,k,m,matGet(mu,k,m)+stepsize*vecGet(dmu,k));
                matSet(sig,k,m,matGet(sig,k,m)*exp(stepsize*vecGet(dsig,k)));
            }
        }
    }
//  update u and Kern
    for (k = 0; k < K; k++)
    {
        vecSet(u,k,0);
        for (m = 0; m < M; m++)
        {
            vecSet(u,k,vecGet(u,k)+matGet(mu,k,m));
        }
        vecSet(u,k,vecGet(u,k)/M);
    }
    zeros(Kern,0);
    int k1, k2;
    for (k1 = 0; k1 < K; k1++)
    {
        for (k2 = 0; k2 < K; k2++)
        {
            temp = 0;
            for (m = 0; m < M; m++)
            {
                if (k1 == k2) temp = temp + matGet(sig,k1,m);
                temp = temp + (matGet(mu,k1,m)-vecGet(u,k1))*(matGet(mu,k2,m)-vecGet(u,k2));
            }
            matSet(Kern,k1,k2,temp/M);
        }
    }
    gsl_vector_free(invKmu);
    gsl_vector_free(AdivB);
    gsl_vector_free(expMuV);
    gsl_matrix_free(invKern);
    gsl_vector_free(dmu);
    gsl_vector_free(dsig);
    gsl_vector_free(colmu);
    gsl_vector_free(colsig);
    return Lbound;
}

// ********************* KMEANS INITIALIZATION ***********************
void centInit(gsl_matrix * X, gsl_matrix * C, int K, int D, int N)
// Initialize centroids
{
    gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(r,0);
    int pick, k, d;
    for (k = 0; k < K; k++)
    {
        pick = ceil(N*gsl_rng_uniform(r));
        for (d = 0; d < D; d++)
        {
            gsl_matrix_set(C,d,k,gsl_matrix_get(X,d,pick));
        }
    }
    gsl_rng_free(r);
}

void indUpdate(gsl_matrix * X, gsl_matrix * C, int indicator[], int K, int D, int N, int pnorm)
// Update latent indicators
{
    int n, k, d, idx;
    double dist[K];
    double val;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < K; k++)
        {
            dist[k] = 0;
            for (d = 0; d < D; d++)
            {
                val = matGet(X,d,n) - matGet(C,d,k);
                if (val >= 0)
                    dist[k] = dist[k] + pow(val,pnorm);
                else
                    dist[k] = dist[k] + pow(-1*val,pnorm);
            }
        }
        idx = 0;
        for (k = 1; k < K; k++)
        {
            if (dist[k] < dist[idx])
                idx = k;
        }
        indicator[n] = idx;
    }
}

void centUpdate(gsl_matrix * X, gsl_matrix * C, int count[], int indicator[], int K, int D, int N)
// Update centroids
{
    int k, d, n;
    for (k = 0; k < K; k++)
        count[k] = 0;

    zeros(C,0);
    for (n = 0; n < N; n++)
    {
        count[indicator[n]]++;
        for (d = 0; d < D; d++)
        {
            gsl_matrix_set(C,d,indicator[n],gsl_matrix_get(C,d,indicator[n]) + gsl_matrix_get(X,d,n));
        }
    }
    for (k = 0; k < K; k++)
    {
        if (count[k] > 0)
        {
            for (d = 0; d < D; d++)
            {
                gsl_matrix_set(C,d,k,gsl_matrix_get(C,d,k)/count[k]);
            }
        }
    }
}

void Kmeans_init(corpus * allDocs, gsl_matrix * Gam, int pnorm, double gamma, int numIte)
{
    int i, d, k, n, w, wordIdx, wordCnt;
    int K = Gam->size1;
    int D = Gam->size2;
    int N = allDocs->numDocs;
    gsl_matrix * Cent = gsl_matrix_alloc(D,K);
    gsl_matrix * X = gsl_matrix_alloc(D,N);
    zeros(X,0);
    for (n = 0; n < N; n++)
    {
        for (w = 0; w < allDocs->docs[n].numUnique; w++)
        {
            wordIdx = vecGet(allDocs->docs[n].wordIdx,w);
            wordCnt = vecGet(allDocs->docs[n].wordCnt,w);
            matSet(X,wordIdx,n,matGet(X,wordIdx,n)+wordCnt);
        }
    }
    double val;
    for (n = 0; n < N; n++)
    {
        val = 0;
        for (d = 0; d < D; d++)
            val = val + matGet(X,d,n);
        for (d = 0; d < D; d++)
            matSet(X,d,n,matGet(X,d,n)/val);
    }
    centInit(X,Cent,K,D,N);

//  main Kmeans loop
    int indicator[N];
    int count[K];
    printf("Beginning Kmeans initialization - this may take a few minutes\n");
    for (i = 0; i < numIte; i++)
    {
        printf("In iteration %d/%d\n",i+1,numIte);
        indUpdate(X,Cent,indicator,K,D,N,pnorm);  // update latent indicators
        centUpdate(X,Cent,count,indicator,K,D,N); // update centroids
    }
    printf("\n");
    for (d = 0; d < D; d++)
    {
        for (k = 0; k < K; k++)
        {
            matSet(Gam,k,d,gamma/D + gamma*matGet(Cent,d,k));
        }
    }
    gsl_matrix_free(Cent);
    gsl_matrix_free(X);
}
