import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

#Define the function being evaluated
func = lambda x,y: np.sin(x**2) * np.cos(y**2)

#Define the bounds of the function and the true value, error
upper_bound = 1
lower_bound = 0
true_value, error = integrate.dblquad(func, lower_bound, upper_bound, lambda x: lower_bound, lambda x: upper_bound)
print(true_value)

#Define the Monte Carlo 2D function
def monte_carlo_2D(x,y):
    n_x = len(x)
    F_i = 0
    for i, j in zip(x, y):
        F_i = F_i + func(j, i)
    return F_i / n_x

#Create vector to hold result and a range of numbers for creating different sizes of Unif(0,1) observations
num = list(range(10, 10000, 100))
result_store = []

#Loop through the range of observation sizes to create uniforms for x and y, apply Monte Carlo integration to them and append to list
for i in num:
    unif_x = np.random.uniform(low=lower_bound, high=upper_bound, size=i)
    unif_y = np.random.uniform(low=lower_bound, high=upper_bound, size=i)
    result = monte_carlo_2D(unif_x, unif_y)
    result_store.append(result)

#Create plots of the results and apply a polynomial fit the plot
plt.plot(num, result_store)
coefficients = np.polyfit(num, result_store, 15)
poly_function = np.poly1d(coefficients)
x_fit = np.linspace(min(num), max(num))
y_fit = poly_function(x_fit)
plt.plot(x_fit, y_fit, color = 'red')
plt.axhline(y = true_value, color = 'black')
plt.xlabel('N')
plt.ylabel('Monte Carlo Result')
plt.title('Monte Carlo Integration Simulating Different Values of N')
plt.grid(True)
plt.show()

#Create error list, logs for evaluating convergence rate
error_result = []
N_log = np.logspace(1, 5, 20)
int_log = N_log.astype(int)
O_rate = [1/np.sqrt(x) for x in int_log]

#Loop through logs and append error calculation to new list
for i in int_log:
    unif_x = np.random.uniform(low = 0, high = 1, size = i)
    unif_y = np.random.uniform(low=0, high=1, size=i)
    new_result = monte_carlo_2D(unif_x, unif_y)
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
plt.show()

#Calculating the Sample Variance, Standard Error, and Confidence Interval @ 100,000

#Generate Samples, randoms, functions
N = 100000
X = np.random.uniform(low = lower_bound, high = upper_bound, size = N)
Y = np.random.uniform(low = lower_bound, high = upper_bound, size = N)
f_X = func(X,Y)

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

plt.show()

