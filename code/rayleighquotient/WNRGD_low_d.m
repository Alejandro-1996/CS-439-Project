gradient_noise_variance = 2;

A = [1.0000    2.0000    4.5000;
    2.0000    3.0000   -1.0000;
    4.5000   -1.0000   -6.0000]

% initial point
x = [1;0;0];

% lists to keep track of the iterates
X = [1];
Y = [0];
Z = [0];

% run gradient descent with step size eta
b = 20; % try 20 and 200 -- doesnt do very well with 200, not surprisingly though
stepsizes = [1/b];
next_gradient = gradFcn(x, A, gradient_noise_variance);
for iter = 2:100
    x = Retr(x, -1/b*next_gradient);
    
    next_gradient = gradFcn(x, A, gradient_noise_variance);
    
    b = b + sum(next_gradient.^2)/b;
    
    % store iterates and step sizes
    X(iter) = x(1);
    Y(iter) = x(2);
    Z(iter) = x(3);
    stepsizes(iter) = 1/b;
end

plot(stepsizes)

% plot iterates
plot3(X, Y, Z,'-o','Color','b','MarkerSize',10,'MarkerFaceColor','#D9FFFF')
hold on
sphere

% plot the critical points (eigenvectors of A)
plot3(-[-0.4501 -0.5903 0.6701], -[0.1550  0.6873 0.7096], -[0.8794 -0.4233 0.2179],'o','Color','b','MarkerSize',10,'MarkerFaceColor','#000000')
hold off
axis equal

% plot(stepsizes)