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

#ifndef IMPORTDATA_H_INCLUDED
#define IMPORTDATA_H_INCLUDED

int countVecSize(char * filename);
// get the dimensionality of the vector written in filename

int countMatRows(char * filename);
// get the number of rows in the matrix written in filename (csv file)

int countMatCols(char * filename);
// get the number of columns in the matrix written in filename (csv file)

void readVector(char * filename, gsl_vector * y);
// read in a vector from filename

void readMatrix(char * filename, gsl_matrix * X);
// read in a matrix from filename (csv file)

void writeVector(char * filename, gsl_vector * y);
// write the vector y to filename

void writeMatrix(char * filename, gsl_matrix * X);
// write the matrix X to filename (as csv file)

void writeArray(char * filename, double y[], int N);
// write the array y to filename

// struct to contain document information
typedef struct doc {
    int numWords;
    int numUnique;
    gsl_vector * wordCnt;
    gsl_vector * wordIdx;
} doc;

// struct to contain the corpus (contains pointers to all docs)
typedef struct corpus {
    doc * docs;
    int numDocs;
    int numWords;
    int vocabSize;
} corpus;

int countDocs(char * filename);
// return the number of documents in filename

void countText(char * filename, corpus * data);
// allocate space for the text data in filename

void readText(char * filename, corpus * data);
// read in text data from filename

#endif // IMPORTDATA_H_INCLUDED
