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

classdef JaggiAtomGenerator < handle
    %ATOMGENERATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess = 'private')
        n; % dimension associated to an nxn matrix
        num_unique_atoms;  % number of unique atoms in the generator
        lambda;
    end
    
    methods(Access = 'public')
        function obj = JaggiAtomGenerator(n, lambda)
            obj.n = n;
            obj.lambda = lambda;
            obj.num_unique_atoms = n*n;
        end
        
        function atom = genAtom(obj, atom_id)
            [i, j] = ind2sub([obj.n, obj.n], atom_id);
            atom = obj.genFromIndex(i, j);
        end
        
        function [atom, atom_id] = getAtom(obj, i, j)
            atom = obj.genFromIndex(i,j);
            atom_id = sub2ind([obj.n obj.n], i, j);
        end
        
        function [atom, atom_id] = randAtom(obj)
            atom_id = randi(obj.num_unique_atoms);
            atom = obj.genAtom(atom_id);
        end
        
    end
    
    methods(Access = 'private')
        function [atom] = genFromIndex(obj, i, j)
            if i==j % is an element on the diagonal
                atom = sparse(i, j, obj.lambda, obj.n, obj.n);
            elseif i < j % negative correlation
                atom = sparse([i i j j]', [i j i j]', [obj.lambda -obj.lambda -obj.lambda obj.lambda]', obj.n, obj.n);
            else % positive correlation
                atom = sparse([i i j j]', [i j i j]', [obj.lambda obj.lambda obj.lambda obj.lambda]', obj.n, obj.n);
            end
        end
    end
    
end
