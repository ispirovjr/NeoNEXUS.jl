using NeoNEXUS
using FFTW

function centeredGrid(N, L=1.0)
    # Centered symmetric grid for periodicity: x[i] + x[N-i+1] = 0
    dx = L/N
    x = range(-L/2 + dx/2, L/2 - dx/2; length=N)
    return x
end


const N = 32
x = centeredGrid(N)
y = centeredGrid(N)
z = centeredGrid(N)

X = reshape(x, :, 1, 1)
Y = reshape(y, 1, :, 1)
Z = reshape(z, 1, 1, :)

# Fourier wave numbers (consistent with FFT ordering)
L = 1

kr = fftfreq(N) .* N .* 2π / L
kx = [kx for kx in kr, ky in kr, kz in kr]
ky = [ky for kx in kr, ky in kr, kz in kr] # this works, but grows in memory and we might want to change indexing in the future
kz = [kz for kx in kr, ky in kr, kz in kr]

field = X.^2 .+ Y*0 .+ Z*0

krr = collect(kr)

wall = NeoNEXUS.SheetFeature(size(field),krr,krr,krr)


testCache = NeoNEXUS.HessianEigenCache(N,N,N)
wall(field,testCache,NeoNEXUS.Write);

testCache

size(wall.significanceMap)


# ERROR CAUSED BY USING FEATURE.KX, WHICH IS A 1D ARR AND NOT WHAT BRAM DOES WITH 3D ARRS
# TODO - MAKE THE HESSIAN SYSTEM WORK WITH 1D ARRAYS INSTEAD TO MAKE THINGS RUN SMOOTHLY AND CLEANLY