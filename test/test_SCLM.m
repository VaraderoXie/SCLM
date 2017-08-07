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

function [] = test_SCLM()
% TEST_SCLM Summary of this function goes here

    %% Load CNAE-9 dataset
    dataset = load('./data/cnae9.mat');
    
    A = sparse(dataset.X);
    Y = dataset.Y;
    
    A_test = sparse(dataset.X_validate);
    Y_test = dataset.Y_validate;
    
    clear dataset;
    
    %[~, Y] = find(Y');
    [p, ~] = size(A);
    
    %% Set tunable options
    options = struct();
    options.num_groups = 2;
    options.lambda_local = 10;
    options.lambda_global = 100;
    options.num_features = p;
    options.batch_size = 20;
    options.iter_max = 100;
    options.num_friends = 3;
    options.num_imposters = 8;
    
    % some options for generation of group IDs
    cluster_min_size = 20;      % minimum number of samples per cluster
    max_attempts = 20;          % number of clustering attempts
    group_metric = 'cosine';    % metric for clustering algorithm
    
    [~, n_train] = size(A);
    [~, n_test] = size(A_test);
    
    A_total = [A A_test];
    all_group_ids = cluster_data(A_total, cluster_min_size, max_attempts, options.num_groups, group_metric);
    options.group_ids = all_group_ids(1:n_train);
    test_group_ids = all_group_ids(n_train+1:n_train+n_test);
    %options.group_ids = cluster_data(A, cluster_min_size, max_attempts, options.num_groups, group_metric);
    
    % Initialize the SCLM algorithm and solve(minimize objective,
    % AKA "train" the classifier)
    sclm = SCLM(options);
    hist = sclm.solve(A, Y);%, triplets);
    
    %% Make predictions on the test set, and evaluate the quality of the predictions
    Y_predict = sclm.predict(A_test, test_group_ids, 3);
    stats = confusionmatStats(Y_test, Y_predict);
    [micro, macro] = micro_macro(stats.confusionMat);
    fprintf('Micro-averaged F1-score: %d\n', micro.fscore);
    
    
    %%  Create a simple plot demonstrating the objective value and number of active constraints
    figure;
    hold on;
    
    subtightplot(1, 2, 1, [0.2, 0.2]);
    plot(hist.f_t);
    title('Objective Function');
    ylabel('Function Value');
    xlabel('Iteration');
    
    subtightplot(1, 2, 2, [0.2, 0.2]);
    plot(hist.num_active);
    title('# Active Constraints');
    ylabel('# Active Contraints');
    xlabel('Iteration');
    
    
    hold off;
    
    % create plot showing sparsity pattern between separate local metrics
    plot_metrics(hist);
    
end


function [top_terms] = best_terms(metrics, terms)

    top_terms = cell(length(metrics),1);

    atoms = metrics{1}.atoms;
    weights = metrics{1}.atom_weights;
    [~, inds] = sort(weights, 'descend');
    top_terms{1}.weights = weights(inds);
    for i=1:length(inds)
        [r, ~, ~] = find(atoms{inds(i)});
        top_terms{1}.terms{i} = terms{r};
    end
    
    for i=2:length(metrics)
        atoms = metrics{i}.atoms;
        weights = metrics{i}.atom_weights;
        [~, inds] = sort(weights, 'descend');
        top_terms{i}.weights = weights(inds);
        for j=1:length(inds)
            [r, ~, ~] = find(atoms{inds(j)});
            if length(r) == 1
                term = terms{r};
            else
                term = sprintf('%s/%s', terms{r(1)}, terms{r(2)});
            end
            top_terms{i}.terms{j} = term;
        end
    end
    
%     atoms = metrics{2}.atoms;
%     weights = metrics{2}.atom_weights;
%     
%     [~, inds] = sort(weights);
%     for i=1:length(inds)
%         [r, ~, ~] = find(atoms{inds(i)});
%         t1 = terms{r(1)};
%         t2 = terms{r(2)};
%     end
    
end

function [] = plot_metrics(hist)

    num_metrics = length(hist.metrics);
    
    shared = logical(hist.metrics{1});
    for i=2:num_metrics
        shared = shared & logical(hist.metrics{2});
    end
    
    plot_colors = {'g', 'c', 'm', 'y', 'r'};
    
    figure;
    for i=1:num_metrics
        
        exclusive = xor(logical(hist.metrics{i}), shared) & logical(hist.metrics{i});
        
        subtightplot(1, num_metrics, i, [0.1, 0.1]);
        spy(shared, 'b', 10);
        hold on;
        spy(exclusive, plot_colors{i}, 8);
        title_str = sprintf('Metric %d sparsity pattern', i);
        title(title_str);
        xlabel('Input #');
        ylabel('Input #');
        hold off;
    end
end

function [cluster_idx, success] = cluster_data(A, cluster_min_size, max_attempts, num_groups, group_metric)

    success = false;
    attempt_count = 1;
    while attempt_count < max_attempts && success == false
        if strcmp(group_metric, 'sqeuclidean')
            [cluster_idx, cluster_centers] = litekmeans(A', num_groups, 'Distance', 'sqeuclidean');
        elseif strcmp(group_metric, 'cosine')
            [cluster_idx, cluster_centers] = litekmeans(A', num_groups, 'Distance', 'cosine');
        else
            error('Invalid group_metric distance metric given!');
        end

        min_size = min_cluster_size(cluster_idx);
        if min_size > cluster_min_size
            success = true;
        end

        attempt_count = attempt_count + 1;
    end
    
    if attempt_count == max_attempts
        error('clustering failure!');
    end

end

function [min_size] = min_cluster_size(cluster_idx) 

    min_size = 9999999999999;
    unique_vals = unique(cluster_idx);
    for i=1:length(unique_vals)
        temp_size = sum(cluster_idx == unique_vals(i));
        if temp_size < min_size
            min_size = temp_size;
        end
    end
end
