classdef DiagMetric < handle
    %DIAGMETRIC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        options = struct();
        atom_gen;
        atoms = cell(0);
        atom_weights = [];
        atom_ids = [];
    end
    
    methods(Access = 'public')
        function [obj] = DiagMetric(options)
           obj.options = options;
           obj.atom_gen = DiagAtomGenerator(obj.options.num_features, obj.options.lambda_global);
        end
        
        function [atom_update] = initial_random(obj)
            % initialize to random atom
            [atom, atom_id] = obj.atom_gen.randAtom();
            obj.atoms{1} = atom;
            obj.atom_weights(1) = 1;
            obj.atom_ids(1) = atom_id;
            
            % calculate impact of initialization on constraint values
            atom_update = obj.atoms{1} * obj.atom_weights(1);
        end
        
        function [cvals]  = initial_update(obj, cvals, XY, XZ)
            % initialize to random atom
            [atom, atom_id] = obj.atom_gen.randAtom();
            obj.atoms{1} = atom;
            obj.atom_weights(1) = 1;
            obj.atom_ids(1) = atom_id;
            
            % calculate impact of initialization on constraint values
            atom_update = obj.atoms{1} * obj.atom_weights(1);
            cvals = cvals + obj.cvals_diff(XY, XZ, atom_update);
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
        
        function [atom_update] = diag_update(obj, step, G)
            G = sparse(1:obj.options.num_features, 1:obj.options.num_features, G, obj.options.num_features, obj.options.num_features);
            
            [s_fw, id_fw] = obj.mexLMO_diag(G, obj.options.lambda_global);
            [id_away] = obj.f_Away(G);
            
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
        end
        
        function [cvals] = update(obj, step, cvals, XY, XZ)
            
            G = obj.f_grad_diag_est(cvals, XY, XZ);
            G = sparse(1:obj.options.num_features, 1:obj.options.num_features, G, obj.options.num_features, obj.options.num_features);
            
            [s_fw, id_fw] = obj.mexLMO_diag(G, obj.options.lambda_global);
            [id_away] = obj.f_Away(G);
            
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
            cvals = cvals + obj.cvals_diff(XY, XZ, atom_update);
            
        end
    end
    
    methods(Access = 'private')
        
        function [atom, atom_id] = mexLMO_diag(obj, G, lambda)
            [i,j] = LMO_diag(G, lambda);
            atom_id = i;
            atom = obj.atom_gen.genAtom(atom_id);
        end
        
        function [G] = f_grad_diag_est(obj, cvals, XY, XZ)
            G_coeff = smooth_hinge_coeff(cvals);
            inds = find(G_coeff ~= 0);
            
            if length(inds) < obj.options.batch_size
                rand_idx = randperm(length(inds), length(inds));
            else
                rand_idx = randperm(length(inds), obj.options.batch_size);
            end

            inds = inds(rand_idx);

            G_coeff = G_coeff(inds);
            Alpha = sparse(1:length(inds), 1:length(inds), G_coeff, length(inds), length(inds));

            XYs = XY(:,inds);
            XZs = XZ(:,inds);

            G = sum((XYs*Alpha).*XYs,2) - sum((XZs*Alpha).*XZs,2);
            G = G./length(inds);
        end
        
        function [atom_id] = f_Away(obj, grad)

            best_val = -Inf;
            atom_id = -1;
            for i=1:length(obj.atoms)
                val = inner_frob(grad, obj.atoms{i});
                if val > best_val
                    best_val = val;
                    atom_id = obj.atom_ids(i);
                end
            end

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
    end
    
end

