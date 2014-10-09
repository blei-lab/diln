-----------------------------------------------------------------------
The Discrete Infinite Logistic Normal (with HDP option) in C
-----------------------------------------------------------------------

(C) Copyright 2010, John Paisley, Chong Wang and David Blei

Written by John Paisley, jpaisley@princeton.edu.

This file is part of DILN-C

DILN-C is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free 
Software Foundation; either version 2 of the License, or 
(at your option) any later version.

DILN-C is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public 
License for more details.

You should have received a copy of the GNU General Public License 
along with this program; if not, write to the Free Software Foundation,  
Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

-----------------------------------------------------------------------

This is a C implementation of the discrete infinite logistic normal (DILN) 
for topic modeling. Variational Bayes is used for inference. 

The hierarchical Dirichlet process (HDP) is also a model option.

In both model priors, the top-level is represented as a stick-breaking
Dirichlet process, and each second-level probability distribution is 
represented as the normalization of a sequence of gamma random variables.

This code requires the GSL, http://www.gnu.org/software/gsl/

-----------------------------------------------------------------------

TABLE OF CONTENTS


A. COMPILING

B. DATA FORMAT

C. TRAINING ON A CORPUS

D. OUTPUT

E. FILES INCLUDED

-----------------------------------------------------------------------

A. COMPILING

Type "make" in a shell. You will need to change the Makefile to
point to the GSL on your machine.


B. DATA FORMAT ********************************************************

This code uses the same data format as in CTM-C by David M. Blei.
A data file contains an entire corpus for training. Each line of a
data file represents a document as follows:

    [M] [term_1]:[count_1] [term_2]:[count_2] ...  [term_N]:[count_N]

[M]: The number of unique terms in the document

[term_i]: An integer associated with the i-th term in a vocabulary.

[count_i]: The number of times that the i-th term appears in the document.

Notes: [count_i] [term_i+1] are separated by a space. Only terms with 
counts greater than zero should be included.


C. TRAINING ON A CORPUS ************************************************

Below is a list of inputs to DILNtm.exe

Command Line: DILNtm.exe argv[1] argv[2] argv[3] argv[4] argv[5] (optional)

argv[1] : corpus file
argv[2] : number of topics (must be > 2)
argv[3] : method (1 = DILN, 2 = HDP)
argv[4] : if argv[4] integer -> number of iterations
          if 0 < argv[4] < 1 -> error threshold (fractional change in bound)
argv[5] : Dirichlet base concentration parameter
          default = 0.5*|Vocab| -> Dir(0.5,...,0.5)

We currently do not provide the ability to do testing.


D. OUTPUT **************************************************************

The code outputs parameter values into individual csv files. The list of output
parameters are given below (output files are [name].txt). (*) indicates that 
these parameters are not output for HDP.

--- Below, each column is a document and each row is a topic ---

A:    matrix of posterior gamma parameters (first parameter)
B:    matrix of posterior gamma parameters (second parameter)
*mu:  matrix of log-normal vector posterior means (doc specific)
*sig: matrix of log-normal vector posterior variances (doc specific)

    --------------------------------------------------------

*u:     posterior mean of log-normal vectors
*Kern:  posterior covariance matrix (kernel) for log-normal vectors
V:      top-level stick-breaking proportions
Gam:    posterior of topics. each row is a topic. each col is a word
Lbound: lower bound as a function of iteration
alpha:	top-level scaling parameter
beta: 	second-level scaling parameter


E. FILES INCLUDED *******************************************************

main.c
DILNfunctions.c (.h) : functions specific to DILN (HDP) inference
gsl_wrapper.c (.h) : wrapper functions to interact with the gsl
importData.c (.h) : functions for importing (and exporting) data

settings.txt : Contains additional initializations and settings not input
in the command line. The default values are:

   alpha_init = 20        (top-level scaling parameter initialization)
   beta_init = 5 	  (second-level scaling parameter initialization)
   bool_learn_alpha = 1   (a boolean indicating whether to learn alpha)
   bool_learn_beta = 0    (a boolean indicating whether to learn beta)
   Kmeans_iterations = 1  (number of Kmeans iterations for initialization)

Makefile : should be changed to point to the GSL on your machine
README.txt : this file
license.txt : gnu license
