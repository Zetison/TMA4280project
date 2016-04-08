close all
clear all
% return
addpath export_fig
files = dir('*.dat');
counter = 2;
for file = files'
    data = load(file.name);
    m = size(data,1);
    n = m+1;
    h = 1/n;
    x = linspace(0,1,n+1);
    y = linspace(0,1,n+1);
    [X,Y] = meshgrid(x,y);
    X = X';
    Y = Y';
    figure(counter)
    surf(X,Y,padarray(data,[1 1]),'EdgeColor','none','LineStyle','none')
    if false %crazy function
        a = 5;
    else
        a = 2.5;
    end
    b = (a-2)/(a+2);
%     j = linspace(0,1,1000);
%     l = @(j) a*(1./(1+b.^j) - 1/2);
    l = @(j) abs(log(a./(j+a/2)-1)/log(b)-2*eps);
%     plot(j,l(j))
    j = jet;
    colormap(jet)
%     colormap([l(j(:,1)) l(j(:,2)) l(j(:,3))])
    camlight
    grid off
%     xlabel x
%     ylabel y
    axis off
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    export_fig(file.name(1:end-4), '-png', '-transparent', '-r400')
    counter = counter + 1;
end

% A = load('../results/convergence_plot_poisson.txt');
% 
% loglog(A(:,1),A(:,2),A(:,1),A(:,1).^2)



