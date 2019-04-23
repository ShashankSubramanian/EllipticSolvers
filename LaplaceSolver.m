% This app script solves the equation \grad k \grad c = f
% with variable coefficient k, preconditioned by constant coefficient
% operators. A generic rhs is set up for this.

clear; clc;

dim = 2;                 % no of dimensions: 2 or 3
n_misc = setup(dim);     % sets up the variable coefficients 
                         % n_misc.k is the variable coefficient
                         % n_misc.c0 is some function

n = n_misc.n;                         
rhs = n_misc.c0;    
dim = n_misc.dim;
if dim == 2
    rhs = reshape(operator(rhs, n_misc), [n n]);
else
    rhs = reshape(operator(rhs, n_misc), [n n n]);
end

[y, flag, relres, iter] = pcg(@(x)operator(x, n_misc), rhs(:), 1E-3, 500, @(x)applyConstCoeffInv(x, n_misc));
fprintf ('PCG: relres: %e, iter: %d\n', relres, iter);

if n_misc.dim == 3
    y = reshape(y, [n n n]);
else
    y = reshape(y, [n n]);
end

save('LaplaceSolver_solution.mat', 'y');

% This function applies the variable laplace operator
function op = operator(x, n_misc)
    k = n_misc.k;
    n = n_misc.n;
    
    if n_misc.dim == 3
        x = reshape(x, [n n n]);
    else
        x = reshape(x, [n n]);
    end

    % Compute gradient
    [gx, gy, gz] = computeGradient(x, n_misc);
    gx = k .* gx;
    gy = k .* gy;
    gz = k .* gz;

    % Compute divergence
    [gx, ~, ~] = computeGradient(gx, n_misc);
    [~, gy, ~] = computeGradient(gy, n_misc);
    [~, ~, gz] = computeGradient(gz, n_misc);
    
    op = gx + gy + gz;

    op = -op(:);
end

% applies the constant coefficient inverse
function op = applyConstCoeffInv(x, n_misc)
    D_inv_pre_compute = -1. * n_misc.D_inv_pre_compute;
    n = n_misc.n;
    if n_misc.dim == 3
        x = reshape(x, [n n n]);
    else
        x = reshape(x, [n n]);
    end
    op = ifftn((fftn(x) ./ D_inv_pre_compute), 'symmetric');
    op = op(:);
end

% This function sets up the problem with variable coefficients computed
% according the the brain geometry
function n_misc = setup(dim)
    % Read the brain geometry
    % Geo: white matter, gray matter, csf, initial tumor concentration
    if dim == 2
        brain = load('BrainGeometry_2D.mat');
    else
        brain = load('BrainGeometry_3D.mat');
    end

    % Create varible coefficients from the brain geometry
    wm = brain.white_matter; gm = brain.gray_matter; 
    csf = brain.csf; 
    c0 = brain.c0;
%     wm = wm .* (1 - c0);
%     gm = gm .* (1 - c0);
%     csf = csf .* (1 - c0);

    kappa = 0.1;     % scaling for the coefficient
    k = kappa * wm;  % variable coefficient vector

    sz = size(k);
    n = sz(1);       % domain size 
    % i * omega for frequency domain
    if (dim == 2)
        i_omega_y = repmat(1i .* [0:n/2-1 0 -n/2+1:-1]', [1 n]);
        i_omega_x = repmat(1i .* [0:n/2-1 0 -n/2+1:-1], [n 1]);
        i_omega_z = 0;
    else
        i_omega_y = repmat(1i .* [0:n/2-1 0 -n/2+1:-1]', [1 n n]);
        i_omega_x = repmat(1i .* [0:n/2-1 0 -n/2+1:-1], [n 1 n]);
        i_omega_z(1,1,:) = 1i .* [0:n/2-1 0 -n/2+1:-1];
        i_omega_z = repmat(i_omega_z, [n n 1]);
    end

    k_avg = mean(k(:));
    D_inv_pre_compute = k_avg .* i_omega_x .* i_omega_x + k_avg .* i_omega_y .* i_omega_y ...
                        + k_avg .* i_omega_z .* i_omega_z;
    
    D_inv_pre_compute(D_inv_pre_compute == 0) = 1;
    
    % General domain info
    n_misc = struct('n', n, 'dim', dim, 'i_omega_x', i_omega_x, 'i_omega_y', i_omega_y, ...
                            'i_omega_z', i_omega_z, 'wm', wm, 'gm', gm, ...
                            'csf', csf, 'c0', c0, 'k', k, 'k_avg', k_avg, ...
                            'D_inv_pre_compute', D_inv_pre_compute);
end
                   
% This function computes the gradient of f using FFTs
function [fx, fy, fz] = computeGradient(f, n_misc)
    i_omega_x = n_misc.i_omega_x;
    i_omega_y = n_misc.i_omega_y;
    i_omega_z = n_misc.i_omega_z;
    
    fx_hat = fft(f, [], 2);
    fx_hat = i_omega_x .* fx_hat;
    fx = ifft(fx_hat, [], 2, 'symmetric');
    
    fy_hat = fft(f, [], 1);
    fy_hat = i_omega_y .* fy_hat;
    fy = ifft(fy_hat, [], 1, 'symmetric');
    
    fz = 0;
    
    if n_misc.dim == 3
        fz_hat = fft(f, [], 3);
        fz_hat = i_omega_z .* fz_hat;
        fz = ifft(fz_hat, [], 3, 'symmetric');   
    end
end

