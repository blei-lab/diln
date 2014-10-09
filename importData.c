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
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "importData.h"
#include "gsl_wrapper.h"

int countVecSize(char * filename)
// get the dimensionality of the vector written in filename
{
    int M = 0;
    int c;
    FILE *file = fopen(filename,"r");
    while ((c = getc(file)) != EOF)
    {
        if (c == '\n')
            M++;
    }
    fclose(file);
    return M;
}

int countMatRows(char * filename)
// get the number of rows in the matrix written in filename (csv file)
{
    int M = 0;
    int c;
    FILE * file = fopen(filename,"r");
    while ((c = getc(file)) != EOF)
    {
        if (c == '\n')
            M++;
    }
    fclose(file);
    return M;
}

int countMatCols(char * filename)
// get the number of columns in the matrix written in filename (csv file)
{
    int N = 0;
    int c;
    FILE * file = fopen(filename,"r");
    while ((c = getc(file)) != '\n')
    {
        if (c == ',')
            N++;
    }
    N++;
    fclose(file);
    return N;
}

void readVector(char * filename, gsl_vector * y)
// read in a vector from filename
{
    double val;
    int i;
    int M = y->size;
    FILE * file = fopen(filename,"r");
    for (i = 0; i < M; i++)
    {
        fscanf(file, "%lf", &val);
        gsl_vector_set(y,i,val);
    }
    fclose(file);
}

void readMatrix(char * filename, gsl_matrix * X)
// read in a matrix from filename (csv file)
{
    int i, c, m = 0, n = 0;
    double val;
    int M = X->size1;
    int N = X->size2;
    FILE * file = fopen(filename,"r");
    for (i = 0; i < M*N; i++)
    {
        fscanf(file, "%lf", &val);
        gsl_matrix_set(X,m,n,val);
        n++;
        if (n == N)
        {
            n = 0;
            m++;
        }
        c = getc(file);
    }
    fclose(file);
}

void writeVector(char * filename, gsl_vector * y)
// write the vector y to filename
{
    FILE * file = fopen(filename,"w");
    int n;
    int N = y->size;
    for (n = 0; n < N; n++)
    {
        fprintf(file,"%lf",gsl_vector_get(y,n));
        putc('\n',file);
    }
    fclose(file);
}

void writeMatrix(char * filename, gsl_matrix * X)
// write the matrix X to filename (as csv file)
{
    FILE * file = fopen(filename,"w");
    int m, n;
    int M = X->size1;
    int N = X->size2;
    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            fprintf(file,"%lf",gsl_matrix_get(X,m,n));
            putc(',',file);
        }
        putc('\n',file);
    }
    fclose(file);
}

void writeArray(char * filename, double y[], int N)
// write the array y to filename
{
    FILE * file = fopen(filename,"w");
    int n;
    for (n = 0; n < N; n++)
    {
        fprintf(file,"%lf",y[n]);
        putc('\n',file);
    }
    fclose(file);
}

int countDocs(char * filename)
// return the number of documents in filename
{
    FILE * file = fopen(filename,"r");
    int D = 0;
    int c;
    while ((c = getc(file)) != EOF)
    {
        if (c == '\n')
            D++;
    }
    fclose(file);
    return D;
}

void countText(char * filename, corpus * data)
// allocate space for the text data in filename
{
    FILE * file = fopen(filename,"r");
    int c;
    int d = 0;
    int numDocUnique = 0;
    data->docs = malloc(sizeof(doc));
    while ((c = getc(file)) != EOF)
    {
        if (c == ':')
            numDocUnique++;

        if (c == '\n')
        {
            data->docs = (doc*) realloc(data->docs, sizeof(doc)*(d+1));
            data->docs[d].numUnique = numDocUnique;
            numDocUnique = 0;
            data->docs[d].wordCnt = gsl_vector_calloc(data->docs[d].numUnique);
            data->docs[d].wordIdx = gsl_vector_calloc(data->docs[d].numUnique);
            d++;
        }
    }
    fclose(file);
}

void readText(char * filename, corpus * data)
// read in text data from filename
{
    FILE * file = fopen(filename,"r");
    int val;
    int d, n, c;
    int totNumWords = 0;
    int docNumWords = 0;
    for (d = 0; d < data->numDocs; d++)
    {
        n = 0;
        fscanf(file, "%d", &val); // remove first term (not of interest)
        while (n < data->docs[d].numUnique)
        {
            fscanf(file, "%d", &val);
            vecSet(data->docs[d].wordIdx,n,val);
            c = getc(file);
            fscanf(file, "%d", &val);
            vecSet(data->docs[d].wordCnt,n,val);
            docNumWords = docNumWords + val;
            totNumWords = totNumWords + val;
            n++;
        }
        data->docs[d].numWords = docNumWords;
        docNumWords = 0;
    }
    data->numWords = totNumWords;
    fclose(file);

    int maxval = 0;
    for (d = 0; d < data->numDocs; d++)
    {
        for (n = 0; n < data->docs[d].numUnique; n++)
        {
            if (vecGet(data->docs[d].wordIdx,n) > maxval)
                maxval = vecGet(data->docs[d].wordIdx,n);
        }
    }
    maxval++;
    data->vocabSize = maxval;
}
