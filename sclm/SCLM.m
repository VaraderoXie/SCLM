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

classdef SCLM < handle
    %SCLM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        hist = struct();
        opts = struct();
        
        diag_metric;
        local_metrics = cell(0);
        
        A;
        Y;
        triplets;
        cvals;
        XY;
        XZ;
        group_ids;
    end
    
    methods(Access = 'public')
        function obj = SCLM(options)
            obj.opts = options;
            obj.group_ids = options.group_ids;
            % initialize the metrics
            obj.diag_metric = DiagMetric(options);
            obj.local_metrics = cell(options.num_groups);
            for i=1:options.num_groups
                obj.local_metrics{i} = LocalMetric(options);
            end
            
        end
        
        function [Y_predict] = predict(obj, X, X_groups, nn)
            if max(X_groups) > max(obj.group_ids)
                err('Input samples have invalid group ids!');
            else
                Y_predict = obj.knn_classify(obj.A, obj.Y, X, obj.group_ids, X_groups, nn);
            end
        end
        
        function [metrics] = get_metrics(obj)
            metrics = cell(obj.opts.num_groups+1,1);
            metrics{1}.atoms = obj.diag_metric.atoms;
            metrics{1}.atom_weights = obj.diag_metric.atom_weights;
            for i=1:obj.opts.num_groups
                metrics{i+1}.atoms = obj.local_metrics{i}.atoms;
                metrics{i+1}.atom_weights = obj.local_metrics{i}.atom_weights;
            end
        end
        
        function [hist] = solve(obj, A, Y)
            obj.A = A;
            obj.Y = Y;
            
            % each local metric generates a different set of local triplet
            % constraints
            for i=1:obj.opts.num_groups
                local_ids = (obj.group_ids == i);
                obj.local_metrics{i}.init_triplets(obj.A(:,local_ids), obj.Y(local_ids), obj.opts.num_friends, obj.opts.num_imposters);
            end
            
            % set each metric to a random starting position
            for i=1:obj.opts.num_groups
                obj.local_metrics{i}.initial_random();
                obj.local_metrics{i}.update_global(obj.diag_metric.initial_random());
            end
            
            % print progress during optimization
            fprintf('%20s%20s%20s%20s\n', 'Iter', 'Ojective', 'Active Constraints', 'Step');

            iter = 1;
            step = 1;
            while iter <= obj.opts.iter_max
                
                % calculate the objective value
                [f_t, num_active] = obj.f_obj();
                
                obj.hist.f_t(iter) = f_t;
                obj.hist.num_active(iter) = num_active;
                
                fprintf('%20d%20d%20d%20d\n', iter, f_t, num_active, step);
                
                step = (2)/(iter + 2);
                
                G_diag = zeros(obj.opts.num_features,1);
                for i=1:obj.opts.num_groups
                    obj.local_metrics{i}.update(step);
                    G_diag = G_diag + obj.local_metrics{i}.diag_grad();
%                     % update local metrics
%                     local_ids = (obj.group_ids == i);
%                     obj.cvals(local_ids) = obj.local_metrics{i}.update(step, obj.cvals(local_ids), obj.XY(:,local_ids), obj.XZ(:,local_ids));
                end
                G_diag = G_diag./obj.opts.num_groups;
                
                % update the global metric using the sampled diagonal
                % gradient from the local metrics
                diag_update = obj.diag_metric.diag_update(step, G_diag);
                for i=1:obj.opts.num_groups
                    % update the constraints on each local metric using the
                    % global update
                    obj.local_metrics{i}.update_global(diag_update);
                end
                
                % update global metric
                %obj.cvals = obj.diag_metric.update(step, obj.cvals, obj.XY, obj.XZ);
                
                iter = iter + 1;
            end
            
            hist = obj.hist;
            hist.metrics = cell(obj.opts.num_groups);
            temp_diag_metric = obj.diag_metric.getMetric();
            for i=1:obj.opts.num_groups
                hist.metrics{i} = temp_diag_metric + obj.local_metrics{i}.getMetric();
            end
            
        end
    end
    
    
    methods(Access = 'private')
        function [f, num_active] = f_obj(obj)
            f = 0;
            num_active = 0;
            for i=1:obj.opts.num_groups
                [f_local, num_active_local] = obj.local_metrics{i}.f_obj();
                f = f + f_local;
                num_active = num_active + num_active_local;
            end
%             C = arrayfun(@obj.smooth_hinge, obj.cvals);
%             num_active = sum(C ~= 0);
%             f = sum(C);
        end
        
        function [Y_predict] = knn_classify(obj, X_train, Y_train, X_test, train_cluster_idx, test_cluster_idx, nn)
            % X_train -- training samples
            % Y_train -- labels of training samples
            % X_test -- testing samples
            % train_cluster_idx -- cluster IDs of X_train
            % test_cluster_idx -- cluster IDs of X_test
            % nn -- # of nearest neighbors to use in classifying test
            % samples
            
            Y_predict = zeros(size(X_test,2),1);
            
            M_global = obj.diag_metric.getMetric();
            M_local = cell(obj.opts.num_groups);
            for i=1:obj.opts.num_groups
                M_local{i} = obj.local_metrics{i}.getMetric();
            end
            
            for i=1:obj.opts.num_groups

               train_idx = find(train_cluster_idx==i);
               test_idx = find(test_cluster_idx==i);

               Xs_train = X_train(:,train_idx);
               Ys_train = Y_train(train_idx);
               Xs_test = X_test(:,test_idx);

               M = M_global + M_local{i};
               f_dist = @(x,y) (x-y)'*M*(x-y);

               Ys = mlknn_classify(Xs_train, Xs_test, Ys_train, f_dist, nn);
               Y_predict(test_idx) = Ys;
            end
        end
    end
    
    methods(Static)
        
        function [y] = smooth_hinge(x)
            if x >= 1
                y = x-0.5;
            elseif x <= 0
                y = 0;
            else
                y = 0.5*x*x;
            end
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

