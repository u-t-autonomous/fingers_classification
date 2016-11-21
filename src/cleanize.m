%Y2 is "cleaned" function
%Algorithm courtesy of "A principal component-based algorithm for denoising in single
%channel data (PCA for denoising in single channel data)". Antonio Mauricio F.L. Miranda de S� et al

function [Y2] = cleanize(sigin)
    
Lag=sigin(1:(length(sigin)-1));
Reg=sigin(2:length(sigin));


Y2=1/sqrt(2)*(Reg-Lag);


