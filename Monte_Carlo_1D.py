import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#Define the function being evaluated
func = lambda x: np.sin(np.pi*x)

#Define the bounds of the function and the true value, error
upper_bound = 1
lower_bound = 0
true_value, error = integrate.quad(func, lower_bound, upper_bound)
print(true_value)

#Define the Monte Carlo 1D function
def monte_carlo_1D(unif_list):

    n = len(unif_list)
    F_i = 0
    for i in unif_list:
        F_i = F_i + func(i)
    return F_i/n

#Create vector to hold result and a range of numbers for creating different sizes of Unif(0,1) observations
new_list = []
num = list(range(10, 10000, 100))

#Loop through the range of observation sizes to create uniforms, apply Monte Carlo integration to them and append to list
for i in num:
    unif_list = np.random.uniform(low = lower_bound, high = upper_bound, size = i)
    result = monte_carlo_1D(unif_list)
    new_list.append(result)

#Create plots of the results and apply a polynomial fit the plot
coefficients = np.polyfit(num, new_list, 15)
poly_function = np.poly1d(coefficients)
x_fit = np.linspace(min(num), max(num))
y_fit = poly_function(x_fit)
plt.plot(num, new_list)
plt.plot(x_fit, y_fit, color = 'red')
plt.axhline(y = true_value, color = 'black')
plt.xlabel('N')
plt.ylabel('Monte Carlo Result')
plt.title('Monte Carlo Integration Simulating Different Values of N')
plt.grid(True)

#Create error list, logs for evaluating convergence rate
error_result = []
N_log = np.logspace(1, 5, 20)
int_log = N_log.astype(int)
O_rate = [1/np.sqrt(x) for x in int_log]

#Loop through logs and append error calculation to new list
for i in int_log:
    unif_list = np.random.uniform(low = lower_bound, high = upper_bound, size = i)
    new_result = monte_carlo_1D(unif_list)
    absolute_error = abs(true_value - new_result)
    relative_error = absolute_error/abs(true_value)
    error_result.append(relative_error)

#Plot Relative Error and Convergence Rate
plt.loglog(int_log, error_result, label = 'Relative Error')
plt.loglog(int_log, O_rate, label = '1 / sqrt(N)')
plt.xlabel('log(N)')
plt.ylabel('log(Relative Error)')
plt.title('Relative Error & 1/sqrt(N)')
plt.legend()
plt.grid(True)

#Calculating the Sample Variance, Standard Error, and Confidence Interval @ 100,000

#Generate Samples, randoms, functions
N = 100000
X = np.random.uniform(low = lower_bound, high = upper_bound, size = N)
f_X = func(X)

#Integral Estimate
I_bar = (upper_bound - lower_bound) * np.mean(f_X)
print(I_bar)
#Calculate the sample variance
s2 = np.var(f_X, ddof=1)
print(s2)
#Calculate the Standard Error
SE = (upper_bound - lower_bound) * np.sqrt(s2/N)
print(SE)
#95% Confidence Interval
z = 1.96
CI_lower = I_bar - z * SE
CI_upper = I_bar + z * SE
print(CI_lower, CI_upper)