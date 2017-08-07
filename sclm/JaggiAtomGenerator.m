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
