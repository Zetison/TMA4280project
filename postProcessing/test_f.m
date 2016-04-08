figure(1)
x = linspace(0,1,1000);
y = linspace(0,1,1000);
[X, Y] = meshgrid(x,y);
X = X';
Y = Y';
a = 3;
b = 10;
c = 18;
d = 14;
g = @(x,y) sin(a*(sin(c*sin(X+Y.*Y))+0.01))+cos(b*(sin(d*Y)+0.01));
surf(X,Y,log10(abs(tan(pi/4*(g(X,Y))))),'EdgeColor','none','LineStyle','none')
% surf(X,Y,g(X,Y),'EdgeColor','none','LineStyle','none')

xlabel x


view(0,90)
