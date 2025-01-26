using LinearAlgebra
using Plots
using ToeplitzMatrices

include("readclassjson.jl")
data = readclassjson("recursive.json")

w = data["w"]
x = data["x"]
y = data["y"]

n = 10
k = 5
sigma = 2
q = 10
mu = 0.1
m = n*k

# a) matrix C

C = zeros(k*n, n)

for i = 1:k*n
    C[i, floor(Int, (i-1)/k) + 1] = 1

end 
display(C)

# b) matrix B

r(j) = exp(-j^2/sigma^2)

row_1 = zeros(n*k)
r_j = [r(j) for j = 0:-1:-q]
row_1[1:length(r_j)] = r_j

col_1 = zeros(n*k)
c_j = [r(j) for j = 0:q]
col_1[1:length(c_j)] = c_j

B = Toeplitz(row_1, col_1)
display(B)

# c) x_reg (regularized least squares)

A = B*C

x_reg = inv(A'*A + mu*I(n))*A'*y # apostrophe = transpose

plot(x, label="x")
plot!(x_reg, label = "x_reg")
savefig("..\\graphics\\Reg_LS.pdf")

# d) Recursive Method

