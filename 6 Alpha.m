% This is a script to compare the Pyhton and Mathematica results regarding
% the computation of alpha (file n°6)

% Define the parameters
R0 = 8.5;
eta_val =  1.8958e8;
mu_val = 2.3446e9;
lambda_t = 3.6;

% Define the function f(x)
f = @(x) -1 + R0 * sqrt((eta_val + x)/mu_val) * cot(R0 * sqrt((eta_val + x)/mu_val)) + (3 * R0)/(2* lambda_t);

% Define the range of x values
x = linspace(-475e4, -450e4, 100000);
% Initialize y array
y = zeros(size(x));

% Compute y values using a loop
for i = 1:length(x)
    y(i) = f(x(i));
end

% Plot the function
figure;
plot(x, y);
xlabel('x');
ylabel('f(x)');
title('Plot of the Function f(x)');
grid on;

% Use fzero to find zeros
x0 = 1; % It's the initial guess
x_zero = fzero(f, x0);

fprintf('Zero of the function f(x) is approximately at x = %.4f\n', x_zero);