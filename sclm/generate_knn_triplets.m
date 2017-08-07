% Generate triplet constraints LMNN style
% Based on code from: Similarity Learning for High-Dimensional Sparse Data,
% Kuan Liu, Aurelien Bellet, Fei Sha

function [Cons, code] = generate_knn_triplets(data, label, nb_nn, nb_imp)

	addpath('helpers');

	[n, d]=size(data);
	Cons=zeros(3,n*nb_nn*nb_imp);

	fprintf('Creating %d triplets (%d-NN, %d imposters)\n',n*nb_nn*nb_imp,nb_nn,nb_imp);

	UL=unique(label);
	
	diff_index=zeros(nb_imp,n);
	for cc = 1:length(UL)
%	    fprintf('%i nearest imposters for class %i\n',nb_imp,UL(cc));
	    i=find(label==UL(cc));
        if(length(i) < nb_nn+1)    % not enough of same label, so repeat a few times
           % should have the effect of creating multiples of same
           % consrtaint, which is OK
           repeats = ceil(nb_nn/length(i));
           i = repmat(i, [repeats+1,1]);
        end
        
	    j=find(label~=UL(cc));
        if(length(j) < nb_imp+1)    % not enough of same label, so repeat a few times
           % should have the effect of creating multiples of same
           % consrtaint, which is OK
           repeats = ceil(nb_imp/length(j));
           j = repmat(j, [repeats+1,1]);
        end
	    NN=LSKnn(data(i,:),data(j,:), 1:nb_imp);
	    diff_index(:,i)=j(NN);
	end

	same_index=zeros(nb_nn,n);
	for cc = 1:length(UL)
% 	   fprintf('%i nearest genuine neighbors for class %i\n',nb_nn,UL(cc));
 	   i=find(label== UL(cc));
       
       if(length(i) < nb_nn+1)    % not enough of same label, so repeat a few times
           % should have the effect of creating multiples of same
           % consrtaint, which is OK
           repeats = ceil(nb_nn/length(i));
           i = repmat(i, [repeats+1,1]);
       end
 	   NN=LSKnn(data(i,:),data(i,:), 2:nb_nn+1);
 	   same_index(:,i)=i(NN);
	end

	clear i j NN;
	
	temp = repmat([1:n],nb_nn*nb_imp,1);
	Cons(1,:)=temp(:);
	temp=zeros(nb_nn*nb_imp,n);
	for i=1:nb_nn
	    temp((i-1)*nb_imp+1:i*nb_imp,:)=repmat(same_index(i,:),nb_imp,1);
	end
	Cons(2,:)=temp(:);
	temp = repmat(diff_index,nb_nn,1);
	Cons(3,:)=temp(:);
		
end

function NN=LSKnn(X1,X2,ks)
	B = 750;
	n = size(X1,1);
	NN=zeros(length(ks),n);
	for i=1:B:n
		BTr = min(B-1,n-i);
		DTr = X2*X1(i:i+BTr,:)';
		[~,nn] = mink(-full(DTr),max(ks));
		NN(:,i:i+BTr) = nn(ks,:);
	end
end

