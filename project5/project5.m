% Mehmet Sinan INCI project 5 for ML in CyberSecurity Course

clear all;
close all;
clc;

test = readtable('test.txt');
train = readtable('train.txt');

data = [train; test]; % fixed the smurf labels in a text editor
labels = data(:,42); % column 42 holds the labels

# use the GUI of Matlab's Classification Learner Toolbox and train over 80% of the data
# be careful to choose the holdout validation