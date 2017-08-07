//   Copyright (C) 2017  Joseph St.Amand

//   This program is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.

//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "matrix.h"
#include "mex.h"
//#include "tbb/task_scheduler_init.h"

#include <cmath>
#include <iostream>
#include <stdint.h>
#include <limits>

/* getValue frunction provided by "Similarity Learning for High-Dimensional Sparse Data" AAAI '15
 * get value corresponding to element (i,j) in sparse matrix
 * i,j start at 1
 * based on binary search
 */
double getValue (mwIndex *Ir, mwIndex *Jc, double *Pr, int i, int j) {

    int k, left, right, mid, cas;
    left = Jc[j-1];
    right = Jc[j]-1;
    while (left <= right) {
        mid = left + (right-left)/2;
        if (Ir[mid]+1 == i)
            return Pr[mid];
        else if (Ir[mid]+1 > i)
            right = mid - 1;
        else
            left = mid + 1;
    }
    return 0.0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //Expected inputs:
    // Sparse matrix G
    // double lambda

    if(nrhs != 2)
        mexErrMsgTxt("Wrong # of input arguments!");

    if(!mxIsSparse(prhs[0]))
        mexErrMsgTxt("First input must be a sparse matrix");

    if(!mxIsDouble(prhs[1]))
        mexErrMsgTxt("Second input must be a double");

    //start by loading in G and lambda
    double* gPr = mxGetPr(prhs[0]);
    mwIndex* gJc = mxGetJc(prhs[0]);
    mwIndex* gIr = mxGetIr(prhs[0]);
    double lambda = mxGetScalar(prhs[1]);
    int nzmax = mxGetNzmax(prhs[0]);
    int rows = mxGetM(prhs[0]);
    int cols = mxGetN(prhs[0]);

    //std::cout << "rows: " << rows << std::endl;
    //std::cout << "cols: " << cols << std::endl;
    //std::cout << "lambda: " << lambda << std::endl;
    double best_score = std::numeric_limits<double>::max();
    double G_val;

    //case where i > j, positives
    int best_i = 1;
    int best_j = 1;

    for(int i = 1; i <= rows; ++i){
        G_val = getValue(gIr, gJc, gPr, i, i);
        double score = G_val*lambda;
        if(score < best_score){
            best_score = score;
            best_i = i;
            best_j = i;
        }
    }
    plhs[0] = mxCreateDoubleScalar((double)best_i);
    plhs[1] = mxCreateDoubleScalar((double)best_j);
    return;
}




/*
 */
