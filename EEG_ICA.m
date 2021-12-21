clear; close all; clc

load('EEG_data')

plot(Data(:,1))
title("Fp1 Before Blink Removal", 'FontSize', 14)

q=16;

[coeff,Data_PCA,latent,tsquared,explained,mu] = pca(Data, 'NumComponents', q);

% compute and display explained variation
disp(strcat("Top ", string(q), " principal components explain ", ...
    string(sum(explained(1:q))), " of variation"))
%% ICA

% compute independent components from principal components
Mdl = rica(Data_PCA, q);
Data_ICA = transform(Mdl, Data_PCA);

%% PLOT RESULTING COMPONENTS

% define number of plots per column of figure
plotsPerCol = 8;

% set up figure
figure(2)
fig = gcf;
fig.Units = 'normalized';
fig.Position = [0 0 1 1];

% plot components
for i = 1:q
    
    subplot(plotsPerCol,ceil(q/plotsPerCol),i)
    plot(Data_ICA(:,i).^2)
    title(strcat("Component ", string(i), " Squared"), 'FontSize', 16)
    ax = gca;
    ax.XTickLabel = {};
    
end
