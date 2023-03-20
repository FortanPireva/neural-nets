x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0


xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b

# relu activation function

y = max(z, 0)

# backward pass

# derivative from next layer - sum of all weights is 1
# backpropagation derivative of the larger function - relu input is 6.0
# 1 if input > 0 else 0
dvalue = 1.0

# derivative of ReLU and the chain rule
drelu_dz = dvalue * (1 if z > 0 else 0)

# partial derivates of the combined, the chain rule, the partial derivative of sum is always 1 regarding the inputs
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1

# partial derivative of the sum operation with respect to each of these,multiplied by the partial derivative
# of the subsequent function ( using the chain rule)  which is ReLU function, denoted by drelu_dz
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db

dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradients on weights
db = drelu_db # gradient on bias ... just 1 here.

# job of optimizer minimize the output, directly apply a negative fractions to this gradient since we want to
# decrease the final output value

print(w,b)

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w,b)

# do another forward pass
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# adding
z = xw0 + xw1 + xw2

# relu activation function
y = max(z,0)

print(y)

# in real nn -s we want to decrease the loss value, with is the last calculation
# in the chain of calculations during the forward pass, and it's the first one to
# calculate the gradient during backpropagation.