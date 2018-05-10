% Mehmet Sinan INCI project 4 for ML in CyberSecurity Course

% Data labels for the weekly data
% The data collected by Google Maps over 5 weeks. The data is taken every half an hour.
% First Row: Timestamp
% Second Row: Latitude
% Third Row: Longitude
% Forth Row: Accuracy
% Fifth Row: Label

clear all;
close all;
clc;

load('data.mat'); % load test and train

classes = [48,49,50,51,52];
trainX = train(:,1:4);
testX = test(:,1:4);
trainY = train(:,5);
testY = test(:,5);
trainY = categorical(trainY,classes);
testY = categorical(testY,classes);

model = mnrfit(trainX,trainY);
% B = mnrfit(X,Y,Name,Value)
% [B,dev,stats] = mnrfit(___)

result = mnrval(model,testX);

% [pihat,dlow,dhi] = mnrval(B,X,stats)
% [pihat,dlow,dhi] = mnrval(B,X,stats,Name,Value)
% [yhat,dlow,dhi] = mnrval(B,X,ssize,stats)
% [yhat,dlow,dhi] = mnrval(B,X,ssize,stats,Name,Value)

plot(result,'DisplayName','pihat'); % showing probability for 5 different locations
title('Location probabilities throughout the week for 30 min time windows');
legend({'48','49','50','51','52'});
xlabel('Time window');
ylabel('Probability of user being in the location');

[argval, argmax] = max(result');
argmax = argmax' + 47;
error_rate = sum(argmax ~= testY) / 336; % Calculating the incorrect labels among test sets
                                         % We first find the # of incorrect labels and then divide
                                         % it with total # of test samples

fprintf('\nTest set error rate is:\t%f', error_rate);
fprintf('\nTest set accuracy is:\t%f', 100-error_rate);
