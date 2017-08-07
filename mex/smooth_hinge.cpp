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

#include "mex.h"
#include <iostream>

inline double smooth_hinge(double x){
    if( x >= 1.0 )
        return x - 0.5;
    else if(x <= 0)
        return 0;
    else
        return 0.5*x*x;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nlhs != 1)
        mexErrMsgTxt("Expecting 1 output!");

    if(nrhs != 1)
        mexErrMsgTxt("Expecting 1 input!");

    mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
    if(ndims != 2)
        mexErrMsgTxt("Expecting input array of dimension 2!");

    mwSize m = mxGetM(prhs[0]);
    mwSize n = mxGetN(prhs[0]);

    if(n != 1){
        mexErrMsgTxt("Input must be a column vector!");
    }

    double* input = (double *)mxGetData(prhs[0]);
    mwSize dims[2] = {m, 1};
    plhs[0] = mxCreateNumericArray(ndims, dims, mxDOUBLE_CLASS, mxREAL);

    if(plhs[0] == NULL)
        mexErrMsgTxt("Could not create mxArray");

    double* output = mxGetPr(plhs[0]);

    for(size_t i = 0; i < m; ++i)
        output[i] = smooth_hinge(input[i]);

    return;
}
