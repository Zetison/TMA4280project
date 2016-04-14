close all
clear all

addpath export_fig
files = dir('postProcessing/*.dat');
counter = 2;
for file = files'
    data = load(['postProcessing/' file.name]);
    m = size(data,1);
    n = m+1;
    h = 1/n;
    x = linspace(0,1,n+1);
    y = linspace(0,1,n+1);
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    figure(counter)
    surf(X,Y,padarray(data.',[1 1]),'EdgeColor','none','LineStyle','none')
    
    j = jet;
    colormap(jet)
    camlight
    grid off
    axis off
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    export_fig(['postProcessing/' file.name(1:end-4)], '-png', '-transparent', '-r400')
    counter = counter + 1;
end


