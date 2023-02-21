module BBclassify

using Pkg
Pkg.add("QuadGK")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
using QuadGK
using CSV
using DataFrames
using LinearAlgebra
using Statistics


# Cronbachs Alpha reliability coefficient
function cba(x)
    covmat = cov(x)
    variance = sum(covmat)
    diag = sum(Diagonal(covmat))
    n = size(covmat)[1]
    (n / (n - 1)) * (1 - (diag / variance))
end

# Livingston and Lewis' Effective Test Length
function etl(mu, sigma, reliability, minimum, maximum)
    ((mu - minimum) * (maximum - mu) - (reliability * sigma)) / (sigma * (1 - reliability))
end

# Lord's k
function k(mu, sigma, reliability, length) 
    sigma_e = sigma * (1 - reliability)
    num = length * ((length - 1) * (sigma - sigma_e) - length * sigma + mu * (length - mu))
    den = 2 * (mu * (length - mu) - (sigma - sigma_e))
    num / den
end

# Beta function
function beta(a, b)
    quadgk(x -> x^(a - 1) * (1 - x)^(b - 1), 0, 1)[1]
end

# Beta distribution probability density function
function dbeta(x, a, b, l, u)
    if x < l || x > u
        0
    else
        (1 / beta(a, b))  * (((x - l)^(a - 1) * (u - x)^(b - 1)) / (u - l)^(a + b - 1))
    end
end

# Binomial distribution probability mass function
function dbinom(p, n, N)
    binomial(N, n) * (p^n * (1 - p)^(N - n))
end

# Lord's two-term approximation to the compound binomial distribution
function dcbinom(p, n, N, k)
    a = dbinom(p, n, N)
    b = dbinom(p, n, N - 2)
    c = dbinom(p, n - 1, N - 2)
    d = dbinom(p, n - 2, N - 2)
    e = k * p * (1 - p)
    a - e * (b - 2*c + d)
end

# Descending factorial function
function dfac(x, r)
    x_o = copy(x)
    for i in 1:length(x)
        if r <= 1
            x[i] = x[i]^r
        else
            for j in 2:r
                x[i] = x[i] * (x_o[i] - j + 1)
            end
        end
    end
    x
end

# True-score distribution moment generating function
function tsm(x, n, k)
    m = [0.0, 0.0, 0.0, 0.0]
    for i in 1:4
        if i == 1
            m[i] = mean(x) / n
        else
            y = copy(x)
            a = mean(dfac(y, i))[1]
            b = dfac([n - 2], i - 2)[1]
            c = 1 / dfac([n], 2)[1]
            m[i] = (a / b) * c
        end
    end
    m
end

# Beta Binomial integration
function bbintegrate1(a, b, l, u, N, n, k, lower, upper, method = "ll")
    if method == "ll"
        quadgk(x -> dbeta(x, a, b, l, u) * dbinom(x, n, N), lower, upper)[1]
    else
        quadgk(x -> dbeta(x, a, b, l, u) * dcbinom(x, n, N, k), lower, upper)[1]
    end
end

# Beta Compound-Binomial integration
function bbintegrate2(a, b, l, u, N, n1, n2, k, lower, upper, method = "ll")
    if method == "ll"
        quadgk(x -> dbeta(x, a, b, l, u) * dbinom(x, n1, N) * dbinom(x, n2, N), lower, upper)[1]
    else
        quadgk(x -> dbeta(x, a, b, l, u) * dcbinom(x, n1, N, k) * dcbinom(x, n2, N, k), lower, upper)[1]
    end
end

# Beta true-score distribution shape and location parameters
function betaparameters(x, n, k, model, l, u)
    m = tsm(x, n, k)
    s2 = m[2] - m[1]^2
    g3 = (m[3] - 3 * m[1] * m[2] + 2 * m[1]^3) / (s2^0.5)^3
    g4 = (m[4] - 4 * m[1] * m[3] + 6 * m[1]^2 * m[2] - 3 * m[1]^4) / (s2^0.5)^4
    if model == 4
        r = 6 * (g4 - g3^2 - 1) / (6 + 3 * g3^2 - 2 * g4)
        if g3 < 0
            a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))^0.5)
            b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))^0.5)
        else
            b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))^0.5)
            a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))^0.5)
        end
        l = m[1] - ((a * (s2 * (a + b + 1))^0.5) / (a * b)^0.5)
        u = m[1] + ((b * (s2 * (a + b + 1))^0.5) / (a * b)^0.5)
    end
    if model == 2
        a = ((l - m[1]) * (l * (m[1] - u) - m[1]^2 + m[1] * u - s2)) / (s2 * (l - u))
        b = ((m[1] - u) * (l * (u - m[1]) + m[1]^2 - m[1] * u + s2)) / (s2 * (u - l))
    end
    Dict("alpha" => a, "beta" => b, "lower" => l, "upper" => u)
end

# Function for calculating classification accuracy and consistency using the Hanson anb Brennan or Livingston and Lewis approaches.
function cac(x, reliability, minimum, maximum, cut, model = 4, lower = 0, upper = 1, failsafe = true, method = "ll", output = ["accuracy", "consistency"])
    out = Dict()
    minimum = Float64(minimum)
    maximum = Float64(maximum)
    cut = Float64.(cut)
    pushfirst!(cut, minimum)
    push!(cut, maximum)
    truecut = Vector{Float64}(undef, size(cut)[1])
    for i in 1:size(truecut)[1]
        truecut[i] = (cut[i] - minimum) / (maximum - minimum)
    end
    if method == "ll"
        x = Float64.(x)
        N_not_rounded = etl(mean(x), var(x), reliability, minimum, maximum)
        N = Int(round(N_not_rounded))
        for i in 1:size(x)[1]       
            x[i] = ((x[i] - minimum) / (maximum - minimum)) * N_not_rounded
        end
        pars = betaparameters(x, N, 0, 4)
        if (failsafe == true && model == 4) && (pars["lower"] < 0 || pars["upper"] > 1)
            pars = betaparameters(x, N, 0, 2, l = lower, u = upper, minimum, maximum, reliability, method)
        end
        pars["etl"] = N_not_rounded
        pars["etl_rounded"] = N
        pars["lords_k"] = 0
        for i in 1:size(cut)[1]
            cut[i] = round(truecut[i] * N)
        end
        cut = Int64.(cut)
    else
        N = Int(maximum)
        K = k(mean(x), var(x), reliability, N)
        pars = betaparameters(x, N, K, 4, minimum, maximum)
        if (failsafe == true && model == 4) && (pars["lower"] < 0 || pars["upper"] > 1)
            pars = betaparameters(x, N, K, 2, l = lower, u = upper)
        end
        pars["atl"] = N
        pars["lords_k"] = K
        cut = Int64.(cut)
    end
    out["Parameters"] = pars
    if "accuracy" in output
        confmat = Array{Float64, 2}(undef, N + 1, size(cut)[1] - 1)
        for i in 1:(size(cut)[1] - 1)
            for j in 1:(N + 1)
                confmat[j, i] = bbintegrate1(pars["alpha"], pars["beta"], pars["lower"], pars["upper"], N, (j - 1), pars["lords_k"], truecut[i], truecut[i + 1], method)
            end
        end
        confusionmatrix = Array{Float64, 2}(undef, size(cut)[1] - 1, size(cut)[1] - 1)
        for i in 1:(size(cut)[1] - 1)
            for j in 1:(size(cut)[1] - 1)
                if i != (size(cut)[1] - 1)
                    confusionmatrix[i, j] = sum(confmat[(cut[i] + 1):cut[i + 1], j])
                else
                    confusionmatrix[i, j] = sum(confmat[(cut[i] + 1):(N + 1), j])
                end
            end
        end
        out["confusion_matrix"] = confusionmatrix
        out["overall_accuracy"] = sum(Diagonal(confusionmatrix))
    end
    if "consistency" in output
        consmat = Array{Float64, 2}(undef, N + 1, N + 1)
        for i in 1:(N + 1)
            for j in 1:(N + 1)
                consmat[i, j] = bbintegrate2(pars["alpha"], pars["beta"], pars["lower"], pars["upper"], N, (i - 1), (j - 1), pars["lords_k"], 0, 1, method)
            end
        end        
        consistencymatrix = Array{Float64, 2}(undef, size(cut)[1] - 1, size(cut)[1] - 1)
        for i in 1:(size(cut)[1] - 1)
            for j in 1:(size(cut)[1] - 1)
                if i == 1 && j == 1
                    consistencymatrix[i, j] = sum(consmat[1:(cut[i + 1]), 1:(cut[j + 1])])
                end
                if i == 1 && (j != 1 && j != size(cut)[1] - 1)
                    consistencymatrix[i, j] = sum(consmat[1:(cut[i + 1]), (cut[j] + 1):(cut[j + 1])])
                end
                if i == 1  && j == size(cut)[1] - 1
                    consistencymatrix[i, j] = sum(consmat[1:(cut[i + 1]), (cut[j] + 1):(cut[j + 1]) + 1])
                end
                if (i != 1 && i != size(cut)[1] - 1) && j == 1
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1]), 1:(cut[j + 1])])
                end
                if (i != 1 && i != size(cut)[1] - 1) && (j != 1 && j != size(cut)[1] - 1)
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1]), (cut[j] + 1):(cut[j + 1])])
                end
                if (i != 1 && i != size(cut)[1] - 1) && j == size(cut)[1] - 1
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1]), (cut[j] + 1):cut[j + 1] + 1])
                end
                if i == size(cut)[1] - 1 && j == 1
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1] + 1), 1:(cut[j + 1])])
                end
                if i == size(cut)[1] - 1 && (j != 1 && j != size(cut)[1] - 1)
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1] + 1), (cut[j] + 1):(cut[j + 1])])
                end
                if i == size(cut)[1] - 1 && j == size(cut)[1] - 1
                    consistencymatrix[i, j] = sum(consmat[(cut[i] + 1):(cut[i + 1] + 1), (cut[j] + 1):(cut[j + 1]) + 1])
                end
            end
        end
        out["consistency_matrix"] = consistencymatrix
        out["overall_consistency"] = sum(Diagonal(consistencymatrix))
    end
    out
end

# Example run with sum-scores.
cac(sumscores, cba(rawscores), 0, 20, [8, 12], 4, 0, 1, true, "ll") #Livingston and Lewis approach
cac(sumscores, cba(rawscores), 0, 20, [8, 12], 4, 0, 1, true, "hb") #Hanson and Brennan approach

# Example run with mean-scores (only works for Livingston and Lewis approach).
cac(meanscores, cba(rawscores), 0, 1, [.4, .6], 4, 0, 1, true, "ll") #


end
