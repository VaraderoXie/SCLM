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

classdef DiagAtomGenerator
    %DIAGATOMGENERATOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(SetAccess = 'private')
        n;
        num_unique_atoms;
        lambda;
    end
    
    methods(Access = 'public')
        function obj = DiagAtomGenerator(n, lambda)
            obj.n = n;
            obj.lambda = lambda;
            obj.num_unique_atoms = n;
        end
        
        function atom = genAtom(obj, atom_id)
            atom = sparse(atom_id, atom_id, obj.lambda, obj.n, obj.n);
        end
        
        function [atom, atom_id] = randAtom(obj)
            atom_id = randi(obj.num_unique_atoms);
            atom = obj.genAtom(atom_id);
        end
    end
end
