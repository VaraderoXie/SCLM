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

classdef LocalMetric < handle
    %LOCALMETRIC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        options = struct();
        triplets = [];
        XY;
        XZ;
        cvals = [];
        atom_gen;
        atoms = cell(0);
        atom_weights = [];
        atom_ids = [];
    end
    
    methods(Access = 'public')
        function [obj] = LocalMetric(options)
            obj.options = options;
            obj.atom_gen = JaggiAtomGenerator(obj.options.num_features, obj.options.lambda_local);
        end
        
        function [M] = getMetric(obj)
            if isempty(obj.atoms)
                M = sparse(obj.options.num_features, obj.options.num_features);
                return;
            else
                M = obj.atoms{1} * obj.atom_weights(1);
                for i=2:length(obj.atoms)
                    M = M + obj.atoms{i} * obj.atom_weights(2);
                end
            end
        end
        
        function [] = init_triplets(obj, A, Y, num_friends, num_imposters)
            obj.triplets = generate_knn_triplets(A', Y, num_friends, num_imposters)';
            obj.cvals = ones(length(obj.triplets),1);
            
            obj.XY = A(:,obj.triplets(:,1)) - A(:,obj.triplets(:,2));
            obj.XZ = A(:,obj.triplets(:,1)) - A(:,obj.triplets(:,3));
        end
        
        function [] = update_global(obj, atom_update)
            % update the constraint values with an update to the global
            % metric
            
            obj.cvals = obj.cvals + obj.cvals_diff(obj.XY, obj.XZ, atom_update);
        end
        
        function [] = initial_random(obj)
            % initialize to random atom
            [atom, atom_id] = obj.atom_gen.randAtom();
            obj.atoms{1} = atom;
            obj.atom_weights(1) = 1;
            obj.atom_ids(1) = atom_id;
            
            % calculate impact of initialization on constraint values
            atom_update = obj.atoms{1} * obj.atom_weights(1);
            obj.cvals = obj.cvals + obj.cvals_diff(obj.XY, obj.XZ, atom_update);
        end
        
        function [cvals] = initial_update(obj, cvals, XY, XZ)
            % initialize to random atom
            [atom, atom_id] = obj.atom_gen.randAtom();
            obj.atoms{1} = atom;
            obj.atom_weights(1) = 1;
            obj.atom_ids(1) = atom_id;
            
            % calculate impact of initialization on constraint values
            atom_update = obj.atoms{1} * obj.atom_weights(1);
            cvals = cvals + obj.cvals_diff(XY, XZ, atom_update);
        end
        
        function [G] = diag_grad(obj)
            G_coeff = smooth_hinge_coeff(obj.cvals);
            inds = find(G_coeff ~= 0);
            
            if isempty(inds) 
                G = zeros(obj.options.num_features,1);
                return;
            elseif length(inds) < obj.options.batch_size
                rand_idx = randperm(length(inds), length(inds));
            else
                rand_idx = randperm(length(inds), obj.options.batch_size);
            end

            inds = inds(rand_idx);

            G_coeff = G_coeff(inds);
            Alpha = sparse(1:length(inds), 1:length(inds), G_coeff, length(inds), length(inds));

            XYs = obj.XY(:,inds);
            XZs = obj.XZ(:,inds);

            G = sum((XYs*Alpha).*XYs,2) - sum((XZs*Alpha).*XZs,2);
            G = G./length(inds);
        end
        
        function [f, num_active] = f_obj(obj)
            C = arrayfun(@obj.smooth_hinge, obj.cvals);
            num_active = sum(C ~= 0);
            f = sum(C);
        end
        
        function [] = update_cvals(obj, cvals_update)
            obj.cvals = obj.cvals + cvals_update;
        end
        
        function [cvals] = update(obj, step)%, cvals, XY, XZ)
            
            %calculate gradient, then solve for forward and away directions
            [G] = obj.f_grad_stoch(obj.XY, obj.XZ, obj.cvals);
            if nnz(G) == 0
                return;
            end
            [s_fw, id_fw] = obj.heuristic_LMO(G);
            [id_away] = obj.f_Away(G, obj.atoms, obj.atom_ids);
            
            % update the solution
            away_index = find(obj.atom_ids == id_away);
            s_away = obj.atoms{away_index};
            max_step = obj.atom_weights(away_index);
            
            if step > max_step
                step = max_step;
            end
            
            % update weight on "away" atom
            if norm(step-max_step) < 1e-6
                % then we need to completely remove an atom
                obj.atom_weights(away_index) = [];
                obj.atom_ids(away_index) = [];
                obj.atoms{away_index} = [];
                obj.atoms(cellfun(@(atoms) isempty(atoms),obj.atoms)) = [];
            else
                obj.atom_weights(away_index) = obj.atom_weights(away_index) - step;
            end

            % update weight on "fw" atom
            if any(obj.atom_ids == id_fw)   % check if FW atom is already in set
                index_fw = find(obj.atom_ids == id_fw);
                obj.atom_weights(index_fw) = obj.atom_weights(index_fw) + step;
            else % introduce a new atom to the active set
                obj.atom_ids = [obj.atom_ids id_fw];
                index = find(obj.atom_ids == id_fw);
                obj.atom_weights(index) = step;
                obj.atoms{index} = s_fw;
            end

            atom_update = step*(s_fw - s_away);
            obj.cvals = obj.cvals + obj.cvals_diff(obj.XY, obj.XZ, atom_update); 
        end
        
    end
    
    methods(Access = 'private')
        
        function [atom_id] = f_Away(obj, G, atoms, atom_ids)
            best_val = -inf;
            for i=1:length(atoms)
                val = inner_frob(G, atoms{i});
                if val > best_val
                    best_val = val;
                    atom_id = atom_ids(i);
                end
            end
        end
                
        function [G, inds] = f_grad_stoch(obj, XY, XZ, cvals)
            G_coeff = arrayfun(@obj.smooth_hinge_G_coeff, cvals);
            
            inds = find(G_coeff ~= 0);
            if isempty(inds) || sum(inds) == 0
                G = sparse(obj.options.num_features, obj.options.num_features);
                return;
            end
            
            if length(inds) > obj.options.batch_size
                rand_inds = randperm(length(inds), obj.options.batch_size);
                inds = inds(rand_inds);
            end
            
            
            G_coeff = G_coeff(inds);
            Alpha = sparse(1:length(inds), 1:length(inds), G_coeff, length(inds), length(inds));
            
            XYs = XY(:,inds);
            XZs = XZ(:,inds);
            
            G = XYs*Alpha*XYs' - XZs*Alpha*XZs';
            G = G./length(inds);
        end
        
        function [atom, atom_id] = heuristic_LMO(obj, G)

            [n, ~] = size(G);

            start_row = randi([1 n], 1, 1);
            [i, j, sign] = LMO_heuristic(G, obj.atom_gen.lambda, start_row);
            if (sign==-1 && i > j) || (sign==1 && i < j)
               a = j;
               j = i;
               i = a;
            end

            [atom, atom_id] = obj.atom_gen.getAtom(i,j);
        end
    end
   
    methods(Static)
        function [diff] = cvals_diff(XY, XZ, update_atom)
            diff = sum((XY)'*update_atom.*(XY)',2) - sum((XZ)'*update_atom.*(XZ)',2);
        end
        
        function [y] = smooth_hinge_G_coeff(x)
            if x >= 1
                y = 1;
            elseif x <= 0
                y = 0;
            else
                y = x;
            end
        end
        
        function [y] = smooth_hinge(x)
            if x >= 1
                y = x-0.5;
            elseif x <= 0
                y = 0;
            else
                y = 0.5*x*x;
            end
        end
    end
    
end

