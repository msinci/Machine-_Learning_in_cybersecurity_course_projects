% Mehmet Sinan INCI project 3 for ML in CSec Course

Xtrain(:,6001) = transpose(Ytrain);
Xtest(:,6001) = transpose(Ytest);

Xtrain(1601:2000,:) = Xtest;
Ytrain(1601:2000) = Ytest;

% new test

n_class = 40;
x_dim = size(Ytrain, 2);
Ycat = zeros(x_dim, n_class);
for i=1:x_dim
    index = Ytrain(i)
    Ycat(i,index) = 1;
end

x_dim = size(Ytest, 2);
Ytest_cat = zeros(x_dim, n_class);
for i=1:x_dim
    index = Ytest(i)
    Ytest_cat(i,index) = 1;
end


x_dim = size(Ytrain, 2);
Ytrain_cat = zeros(x_dim, n_class);
for i=1:x_dim
    index = Ytrain(i)
    Ytrain_cat(i,index) = 1;
end

X_train = X_train';


hiddenSize1 = 1000;

autoenc1 = trainAutoencoder(X_train,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

view(autoenc1)

feat1 = encode(autoenc1,X_train);

%% Training the second autoencoder

hiddenSize2 = 200;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)
feat2 = encode(autoenc2,feat1);

%% Training the final softmax layer
% Train a softmax layer to classify the 50-dimensional feature vectors.
% Unlike the autoencoders, you train the softmax layer in a supervised
% fashion using labels for the training data.

softnet = trainSoftmaxLayer(feat2,Ycat','MaxEpochs',4000);

%%
% As was explained, the encoders from the autoencoders have been used to
% extract features. You can stack the encoders from the autoencoders 
% together with the softmax layer to form a deep network.

deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)

%%
% You can visualize the results with a confusion matrix. The numbers in the
% bottom right-hand square of the matrix give the overall accuracy.

y = deepnet(X_test');
plotconfusion(Ytest_cat',y);


% Fine tuning, re-training the AE with training labels
deepnet = train(deepnet,X_train,Ytrain_cat');

y = deepnet(X_test');
plotconfusion(Ytest_cat',y);

[c,cm] = confusion(Ytest_cat',y)
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

plotroc(Ytest_cat',y)

