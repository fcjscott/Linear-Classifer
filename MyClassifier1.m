
classdef MyClassifier1 < handle
    
    properties (Access = public)
        K                     % Number of classes
        N                     % Number of features
        W                     % Hyperplanes vectors
        w                     % Hyperplane biases
        s                     % Labels to classify (+1 or -1)
        c                     % One vs One classification combination sets
        f                     % Decision values
        voter
        % You can add any extra properties you would like
        
    end
    
        
    methods (Access = public)
        
        function obj = MyClassifier1(K,N)    % Class Constructor
            obj.K = K;
            obj.N = N;
            obj.W = [];
            obj.w = [];
            obj.s = [];
            obj.c = [];                      % Combinatorial matrix
            obj.voter = [];
            obj.f = [];
        end
        
        
        function obj = train(obj,trainData,trainLabel)     
            obj.c = combnk(1:obj.K,2);
            obj.c = obj.c-1;
            for m = 1:length(obj.c);
                obj.s = 2*(trainLabel==obj.c(m,1))-1;
                class1_v = find(obj.s==1);
                class2_v = find(trainLabel==obj.c(m,2));

                for i = 1:round(length(class1_v)/10);
                    k1 = randi(length(class1_v));
                    k2 = randi(length(class2_v));
                    Class1(1:obj.N,i) = trainData(1:obj.N,class1_v(k1));
                    Class2(1:obj.N,i) = trainData(1:obj.N,class2_v(k2));
                end
                New_Train = [Class1 Class2];
                New_s = [ones(length(Class1(1,:)),1);-ones(length(Class2(1,:)),1)];

                [a,b] = SeparatingHyperplane(New_s,New_Train)
                obj.W(:,m) = a;
                obj.w(m) = b;
            end
            %%% THIS IS WHERE YOU SHOULD WRITE YOUR TRAINING FUNCTION
            %
            % The inputs to this function are:
            %
            % obj: a reference to the classifier object.
            % trainData: a matrix of dimesions N x N_train, where N_train
            % is the number of inputs used for training. Each column is an
            % input vector.
            % trainLabel: a vector of length N_train. Each element is the
            % label for the corresponding input column vector in trainData.
            %
            % Make sure that your code sets the classifier parameters after
            % training. For example, your code should include a line that
            % looks like "obj.W = a" and "obj.W = b" for some variables "a"
            % and "b".
             
        end
        
        function [testResults] = classify(obj,testData)
            if (isempty(obj.W) || isempty(obj.w))
                error('Classifier is not trained yet.');
            end
            testResults = zeros(length(testData),1);
            ct = zeros(10,length(testData));
            for i = 1:length(testData)
                for j = 1:length(obj.c)
                    obj.f(j,i) = sign(obj.W(:,j)'*testData(:,i)+obj.w(j));
                    if obj.f(j,i) == -1
                        obj.f(j,i) = obj.f(j,i)+3;
                    end
                        switch obj.c(j,obj.f(j,i))
                            case 0
                                ct(1,i) = ct(1,i)+1;
                            case 1
                                ct(2,i) = ct(2,i)+1;
                            case 2
                                ct(3,i) = ct(3,i)+1;
                            case 3
                                ct(4,i) = ct(4,i)+1;
                            case 4
                                ct(5,i) = ct(5,i)+1;
                            case 5
                                ct(6,i) = ct(6,i)+1;
                            case 6
                                ct(7,i) = ct(7,i)+1;
                            case 7
                                ct(8,i) = ct(8,i)+1;
                            case 8
                                ct(9,i) = ct(9,i)+1;
                            case 9
                                ct(10,i) = ct(10,i)+1;
                        end
                end
%      dealing with voting system if it generates two same maximum
%      value,using "Maximum absolute value" approach
                if length(find(ct(:,i)== max(ct(:,i))))>= 2
                    retrain = find(ct(:,i)== max(ct(:,i)))-1;
                    retrain_c = combnk(retrain,2);
                    [m n] = size(retrain_c);
                    val = 0;
                    for k = 1:m
                        a = find(obj.c(:,1)==retrain_c(k,1));
                        b = find(obj.c(:,2)==retrain_c(k,2));
                        c = intersect(a,b);
                        if isempty(c)
                            break;
                        else
                        val(k) = obj.W(:,c)'*testData(:,i)+obj.w(c);
                        end
                    end 
                    pos_i = find(val==max(val))||find(val==min(val));
                    pos_j = (sign(val(pos_i))==-1)+1;
                    extra_vote = retrain_c(pos_i,pos_j);
                    ct(extra_vote+1,i) = ct(extra_vote+1,i)+1;
                end
           end
           obj.voter = ct;
           max_vote = max(obj.voter);
           for k = 1:length(testData)
               testResults(k,1) = find(obj.voter(:,k)==max_vote(k))-1;
           end 
            %%% THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
            %
            % The inputs to this function are:
            %
            % obj: a reference to the classifier object.
            % testData: a matrix of dimesions N x N_test, where N_test
            % is the number of inputs used for testing. Each column is an
            % input vector.
            %
            % The outputs of this function are:
            %
            % testResults: this should be a vector of length N_test,
            % containing the estimations of the classes of all the N_test
            % inputs.
        end
        
        function [I] = ConstructCorrupted(obj)
            num_corr = round(randi(length(obj.c))/2);
            I = sort(randperm(length(obj.c),num_corr),'descend');
        end
        
        function [testResults] = TestCorrupted(obj,testData,I)
            for i = 1:length(I)
                obj.c(I(i),:) = [];
                obj.W(:,I(i)) = [];
                obj.w(I(i)) = [];
            end          
            testResults = classify(obj,testData);
        end              
    end 
end