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