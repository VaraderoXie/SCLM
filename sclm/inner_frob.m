function [A] = inner_frob(A, B)
%INNER_FROB Summary of this function goes here
%   Detailed explanation goes here
    A = A.*B;
    for i=1:ndims(A)
        A = sum(A);
    end
    A = full(A);
end

