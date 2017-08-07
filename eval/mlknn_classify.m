%    Copyright (C) 2017  Joseph St.Amand

%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.

%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.

%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.

function [Y_predict] = mlknn_classify(X_train, X_test, Y_train, f_dist, nn)
%MLKNN_CLASSIFY Summary of this function goes here
%   X_train -- training instances(have labels)
%   X_test -- instances to produce label predictions for
%   Y_train -- labels that go with X_train instances
%   M -- covariance matrix for mahalabobis distance
%   nn -- number of nearest neighbors for classifier

    [p, n_train] = size(X_train);
    [~, n_test] = size(X_test);
    
    Y_predict = zeros(n_test,1);
    for i=1:n_test
        Y_predict(i) = single_classify(X_train, X_test(:,i), Y_train, f_dist, nn); 
    end
    
end

function [y] = single_classify(X_train, x, Y_train, f_dist, nn)

    [~,n] = size(X_train);
    D = zeros(n,1);
    for i=1:n
        D(i) = f_dist(X_train(:,i),x);
    end
    
    %find the closest samples and select the labels from those samples
    [~, ind] = sort(D);    
    ind = ind(1:nn);
    
    Y_test = Y_train(ind);
    y = mode(Y_test,1);
    
end
