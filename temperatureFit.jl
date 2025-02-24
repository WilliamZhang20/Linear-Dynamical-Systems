using LinearAlgebra
using Random, Distributions
using Plots
using LaTeXStrings

include("readclassjson.jl")
data = readclassjson("mid22_Data/tempdata.json")
x_train = data["x_train"]
y_train = data["y_train"]

scatter(x_train, y_train, label=false, xticks=[0,2,4,6,8,10,12],
title="Daily Maximum Temperature", ylabel="Temperature (F)", xlabel="Month");

n = 25
ms = [3,6,12,24] # numbers of functs
error = []

for m = ms
    fs = [] # builds up a matrix F of the linear affine functions applied to all data points
    for j = 0:m
        function f(x) # each function is piecewise and depends on x & j (the index of the function)
            if x < (j-1)*12/m
                return 0
            elseif x < j * 12 / m
                return m*x/12 - (j-1)
            else
                return 1
            end
        end
    push!(fs, f)
    end

    # Now to form the final fit...
    G = reshape([fs[j](x_train[i]) for j = 1:m+1 for i=1:n], n, m+1)
    alpha = G \ y_train
    y_fit = G * alpha
    push!(error, norm(y_train - y_fit)^2) # error is objective squared
    plot!(x_train, G * alpha, label=LaTeXString("\$m = $m\$"))
end

plot!()
savefig(".\\graphics\\TemperatureFits.pdf")