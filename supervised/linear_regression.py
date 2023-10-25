# %%
import numpy as np
import matplotlib.pyplot as plt

# %%


def generate_linear_data(slope, intercept):

    return


# %%
intercept = 2
slope = 3
x_list = np.arange(0, 100, 5)
eps = np.random.normal(loc=0, scale=10.0, size=len(x_list))
y_list = intercept + slope*x_list + eps

# %%
# Find intercept_star and slope_star based on X & Y
mean_x = np.mean(x_list)
mean_y = np.mean(y_list)
var_x = np.var(x_list, ddof=1)
cov_xy = np.cov(x_list, y_list)[0,1]
slope_s = cov_xy/var_x
intercept_s = mean_y - slope_s*mean_x
y_star = intercept_s + slope_s*x_list

print("intercept star: ", intercept_s)
print("slope star: ", slope_s)

# %%
# Plot
fig, ax = plt.subplots()
ax.scatter(x_list, y_list, alpha=0.5, label="data")

#plot actual line
actual_line, = ax.plot(x_list, intercept + slope*x_list, linestyle='-', color='blue',label='actual line')

# plot fitting line
fitting_line, = ax.plot(x_list, y_star, linestyle='-', color='red',label='fitting line')

ax.legend(loc='lower right')
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
