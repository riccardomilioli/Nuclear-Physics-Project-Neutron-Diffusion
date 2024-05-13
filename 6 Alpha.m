% This is a script to compare the Pyhton and Mathematica results regarding
% the computation of alpha (file n°6)

% Define the parameters
R0 = 8.5; % [cm]
eta_val =  1.8958e8; % [1/s]
mu_val = 2.3446e9; % [cm^2/s]
lambda_t = 3.6; % [cm]

% Define the function f(x)
f = @(x) -1 + R0 * sqrt((eta_val + x)/mu_val) * cot(R0 * sqrt((eta_val + x)/mu_val)) + (3 * R0)/(2* lambda_t);

% Define the range of x values:
x = linspace(-475e4, -450e4, 100000);
% Initialize a new array y:
y = zeros(size(x));

% Fill the array y with the values of f(x):
for i = 1:length(x) % So we make sure that they have the same dimensionality
    y(i) = f(x(i));
end

% Plot the function
figure;
plot(x, y);
xlabel('alpha');
ylabel('f(alpha)');
title('Plot of the Function f(alpha)');
grid on;

% Let's also check where is the zero:
x0 = 1; % It's the initial guess
x_zero = fzero(f, x0);

fprintf('The zero of f is at alpha = %.4f\n', x_zero);
