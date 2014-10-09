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

/**********************************************************
*            DILN Topic Model (with HDP option)
*        models learned using variational inference
*
*   argv[1] : corpus file
*   argv[2] : number of topics (must be > 2)
*   argv[3] : method (1 = DILN, 2 = HDP)
*   argv[4] : if argv[4] integer -> number of iterations
*             if 0 < argv[4] < 1 -> error threshold
*   argv[5] : Dirichlet base concentration parameter
*             default = 0.5*|Vocab| -> Dir(0.5,...,0.5)
*
*   written by: John Paisley
*               Princeton University
*               Department of Computer Science
*               jpaisley@princeton.edu
*
***********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_sf_gamma.h>
#include "importData.h"
#include "gsl_wrapper.h"
#include "DILNfunctions.h"

int MAXITE = 1000;

int main(int argc,char *argv[])
{
    if (argc < 5)
    {
        printf("\n*** List of inputs to DILNtm.exe ***\n\n");
        printf("Command Line: DILNtm.exe argv[1] argv[2] argv[3] argv[4] argv[5] (optional)\n\n");
        printf("argv[1] : corpus file\n");
        printf("argv[2] : number of topics (must be > 2)\n");
        printf("argv[3] : method (1 = DILN, 2 = HDP)\n");
        printf("argv[4] : if argv[4] integer -> number of iterations\n");
        printf("          if 0 < argv[4] < 1 -> error threshold\n");
        printf("argv[5] : Dirichlet base concentration parameter\n");
        printf("          default = 0.5*|Vocab| -> Dir(0.5,...,0.5)\n");
        exit(1);
    }

    int K = atoi(argv[2]);
    if (K < 1) error("Error: Select a positive number of topics");

    if (atoi(argv[3]) != 1 && atoi(argv[3]) != 2) error("Error: Select valid algorithm option: 1 = DILN, 2 = HDP");
    int bool_HDP = 0;
    if (atoi(argv[3]) == 2) bool_HDP = 1;

    int bool_thresh;
    double T;
    if (atof(argv[4]) < 0){
        error("Error: arg[4] must be greater than zero");
    }
    else if (atof(argv[4]) < 1){
        T = atof(argv[4]);
        bool_thresh = 1;
    }
    else{
        T = atof(argv[4]);
        bool_thresh = 0;
    }

//  import data
    corpus * allDocs;
    allDocs = malloc(sizeof(corpus));
    allDocs->numDocs = countDocs(argv[1]);
    countText(argv[1],allDocs);
    readText(argv[1],allDocs);

    printf("\nNumber of documents: %d\n",allDocs->numDocs);
    printf("Total number of words: %d\n",allDocs->numWords);
    printf("Vocabulary size: %d\n\n",allDocs->vocabSize);

    double gamma;
    if (argc > 5)
    {
        gamma = atof(argv[5]);
        if (gamma <= 0)
            error("Error: argv[5] must be greater than zero");
    }
    else gamma = .5*allDocs->vocabSize;

//  allocate space for parameters
    gsl_matrix * A = gsl_matrix_calloc(K,allDocs->numDocs);
    gsl_matrix * B = gsl_matrix_calloc(K,allDocs->numDocs);
    gsl_matrix * mu = gsl_matrix_calloc(K,allDocs->numDocs);
    gsl_matrix * sig = gsl_matrix_calloc(K,allDocs->numDocs);
    gsl_vector * u = gsl_vector_calloc(K);
    gsl_matrix * Kern = gsl_matrix_calloc(K,K);
    gsl_vector * V = gsl_vector_calloc(K);
    gsl_matrix * Gam = gsl_matrix_calloc(K,allDocs->vocabSize);
    gsl_matrix * N = gsl_matrix_calloc(K,allDocs->numDocs);
    double alpha;
    double beta;

//  allocate space for other values of interest
    gsl_vector * p = gsl_vector_calloc(K);

//  read in some additional settings from file
    char * filename = "settings.txt";
    FILE * fileinit = fopen(filename,"r");
    int s;
    char c;
    double tmp;
    int bool_alpha_learn, bool_beta_learn, Kmeans_iterations;
    for (s = 0; s < 5; s++)
    {
        while ((c = getc(fileinit)) != '=');
        fscanf(fileinit, "%lf", &tmp);
        switch (s){
            case 0: alpha = tmp;
            case 1: beta = tmp;
            case 2: bool_alpha_learn = tmp;
            case 3: bool_beta_learn = tmp;
            case 4: Kmeans_iterations = tmp;
        }
    }
    fclose(fileinit);

//  initialize topic posterior parameters with kmeans and top level stick-breaking proportions, V_k
    V_init(V);
    stick_break(p,V);
    zeros(A,10);
    zeros(B,10);
    eye(Kern,1);
    int pnorm = 1;
    Kmeans_init(allDocs, Gam, pnorm, gamma, Kmeans_iterations);
    if (bool_HDP == 0) zeros(sig,1);

//  main loop
    int bool_continue = 1;
    int ite = 0;
    double err;
    double temp;
    double Lbound[MAXITE];
    double Lbound_const = allDocs->numDocs*K/2 + K*gsl_sf_lngamma(gamma) - allDocs->vocabSize*K*gsl_sf_lngamma(gamma/allDocs->vocabSize);
    if (bool_HDP == 0) printf("*** Running variational DILN algorithm ***\n");
    else printf("*** Running variational HDP algorithm ***\n");
    while (bool_continue)
    {
        ite++;
        if (bool_thresh == 0) printf("Iteration %d/%d",ite,(int)T);
        else printf("Iteration %d",ite);
        (ite > 9) ? printf(" :: ") : printf(" ::: ");
        Lbound[ite-1] = Lbound_const;

    //  update Dirichlet parameters and counts matrix and update part of lower bound
        temp = VB_Estep(allDocs,Gam,A,B,N,gamma);
        Lbound[ite-1] = Lbound[ite-1] + temp;

    //  update A and B and part of lower bound
        temp = VB_Mstep_AB(allDocs,A,B,N,beta,p,mu,sig);
        Lbound[ite-1] = Lbound[ite-1] + temp;

    //  update V and part of lower bound
        temp = VB_Mstep_V(A,B,mu,V,p,alpha,beta);
        Lbound[ite-1] = Lbound[ite-1] + temp;

     // update mu, sig, u, Kern and part of lower bound
        if (bool_HDP == 0) // if HDP is selected, this part is not run
        {
            temp = VB_Mstep_lognorm(mu,sig,A,B,p,u,Kern,beta);
            Lbound[ite-1] = Lbound[ite-1] + temp;
        }

    //  update alpha and beta
        if (bool_alpha_learn == 1) alpha = up_alpha(K,V);
        Lbound[ite-1] = Lbound[ite-1] + K*log(alpha);
        if (bool_beta_learn == 1) beta = up_beta(beta,p,mu,A,B);

        if (ite == 1) printf("Lower bound: %.f\n",Lbound[ite-1]);
    //  check whether to terminate algorithm
        if (bool_thresh == 1)
        {
            if (ite > 1)
            {
                printf("Lower bound: %.f ::: ",Lbound[ite-1]);
                err = sqrt(pow((Lbound[ite-1]-Lbound[ite-2])/Lbound[ite-2],2));
                if (err < T || ite == MAXITE) bool_continue = 0;
                printf("Fractional Change: %f\n",err);
            }
        }
        else if (bool_thresh == 0)
        {
            if (ite > 1) printf("Lower bound: %.f\n",Lbound[ite-1]);
            if (ite == (int) T || ite == MAXITE)
                bool_continue = 0;
        }
    }

//  write results to file
    char * f1 = "A.txt";
    writeMatrix(f1,A);
    char * f2 = "B.txt";
    writeMatrix(f2,B);
    if (bool_HDP == 0)
    {
        char * f3 = "mu.txt";
        writeMatrix(f3,mu);
        char * f4 = "sig.txt";
        writeMatrix(f4,sig);
        char * f5 = "u.txt";
        writeVector(f5,u);
        char * f6 = "Kern.txt";
        writeMatrix(f6,Kern);
    }
    char * f7 = "V.txt";
    writeVector(f7,V);
    char * f8 = "Gam.txt";
    writeMatrix(f8,Gam);
    char * f9 = "Lbound.txt";
    writeArray(f9,Lbound,ite);

    char * f10 = "alpha.txt";
    FILE * file = fopen(f10,"w");
    fprintf(file,"%lf",alpha);
    putc('\n',file);
    fclose(file);

    char * f11 = "beta.txt";
    file = fopen(f11,"w");
    fprintf(file,"%lf",beta);
    putc('\n',file);
    fclose(file);

    return 0;
}
