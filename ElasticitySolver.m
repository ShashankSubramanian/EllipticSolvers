% This app script solves the linear elasticity equation preconditioned by
% constant coefficient spectral operators. The variable coefficients are
% read and a generic forcing function is applied for this test-case.

clear; clc;

dim = 2;          % dimension; 2:2D or 3:3D
n_misc = setup(dim); % sets up the variable coefficients
                     % n_misc.mu and n_misc.la are the spatially variable 
                     % Lam`e coefficients

n = n_misc.n;                         
rhs = n_misc.c0;    

% set up the rhs for the elasiticity equation -- some body force 
[fx, fy, fz] = computeForce(rhs, n_misc);
if n_misc.dim == 3
    rhs = [fx(:); fy(:); fz(:)];
else
    rhs = [fx(:); fy(:)];
end

[y, flag, relres, iter] = gmres(@(x)operator(x, n_misc), rhs(:), [],  1E-3, 100, @(x)applyConstCoeffInv(x, n_misc));
fprintf ('GMRES: relres: %e, iter: %d\n', relres, iter(2));

if n_misc.dim == 3
    n_global = n^3;
else
    n_global = n^2;
end

u = y(1:n_global);
v = y(n_global+1:2*n_global);
if n_misc.dim == 3
    w = y(2*n_global+1:3*n_global);
else
    w = 0;
end
        
y = sqrt(u.^2 + v.^2 + w.^2);    

if n_misc.dim == 3
    y = reshape(y, [n n n]);
else
    y = reshape(y, [n n]);
end

save('ElasticitySolver_solution.mat', 'y');

% This function applies the variable elasitcity operator
function op = operator(x, n_misc)
    n = n_misc.n;
    mu = n_misc.mu;
    lam = n_misc.lam;
    r_s = n_misc.rho_screen;
    
    if n_misc.dim == 3
        n_global = n^3;
    else
        n_global = n^2;
    end
    
    u = x(1:n_global);
    v = x(n_global+1:2*n_global);
    if n_misc.dim == 3
        w = x(2*n_global+1:3*n_global);
        u = reshape(u, [n n n]);
        v = reshape(v, [n n n]);
        w = reshape(w, [n n n]);
    else
        u = reshape(u, [n n]);
        v = reshape(v, [n n]);
    end
    
    [dudx, dudy, dudz] = computeGradient(u, n_misc);
    [dvdx, dvdy, dvdz] = computeGradient(v, n_misc);
    if n_misc.dim == 3
        [dwdx, dwdy, dwdz] = computeGradient(w, n_misc);
    else
        dwdx = 0;
        dwdy = 0;
        dwdz = 0;
    end
    
    div_u = dudx + dvdy;
    if n_misc.dim == 3
        div_u = div_u + dwdz;
    end
    
    % grad (lam .* div(u))
    [g_l_x, g_l_y, g_l_z] = computeGradient(lam .* div_u, n_misc);
    
    d_m_u = computeDiv(mu .* (  dudx +   dudx), ...
                mu .* (  dudy +   dvdx), ...
                mu .* (  dudz +   dwdx), ...
                n_misc);
            
    d_m_v = computeDiv(mu .* (  dvdx +   dudy), ...
                mu .* (  dvdy +   dvdy), ...
                mu .* (  dvdz +   dwdy), ...
                n_misc);
         
    if n_misc.dim == 3
        d_m_w = computeDiv(mu .* (  dwdx +   dudz), ...
                    mu .* (  dwdy +   dvdz), ...
                    mu .* (  dwdz +   dwdz), ...
                    n_misc);
    end
    
    op_x = g_l_x + d_m_u - r_s .* u;
    op_y = g_l_y + d_m_v - r_s .* v;
    
    if n_misc.dim == 3
        op_z = g_l_z + d_m_w - r_s .* w;
        op = [op_x(:); op_y(:); op_z(:)];
    else
        op = [op_x(:); op_y(:)];
    end
 
end

% applies the constant coefficient inverse using FFT and sherman morrison
function op = applyConstCoeffInv(x, n_misc)
    n = n_misc.n;
    mu = mean(n_misc.mu(:));
    la = mean(n_misc.lam(:));
    rho_screen_avg = mean(n_misc.rho_screen(:));

    if n_misc.dim == 3
        n_global = n^3;
    else
        n_global = n^2;
    end
    
    % get individual forces
    fx = x(1:n_global);
    fy = x(n_global+1:2*n_global);
    if n_misc.dim == 3
        fz = x(2*n_global+1:3*n_global);
        fx = reshape(fx, [n n n]);
        fy = reshape(fy, [n n n]);
        fz = reshape(fz, [n n n]);
    else
        fx = reshape(fx, [n n]);
        fy = reshape(fy, [n n]);
    end

    if n_misc.dim == 2
        gradx_hat = fftn(fx);
        grady_hat = fftn(fy);

        wx = n_misc.i_omega_x;
        wy = n_misc.i_omega_y;

        % vectorize the w apply using sparse matrices instead of loops
        W = sparse(n^2 * 2, n^2 * 1);
        odd_entries = spdiags(wx(:), 0, n^2, n^2);
        even_entries = spdiags(wy(:), 0, n^2, n^2);
        W(1:2:end,:) = odd_entries;
        W(2:2:end,:) = even_entries;
        
        wx = wx(:); wy = wy(:);
        wTw = wx.^2 + wy.^2;
        temp_mat = [wTw.'; wTw.'];
        wTw = reshape (temp_mat, 1, [])';
        t1 = mu * wTw - rho_screen_avg;
        t1 = 1 ./ t1;
        
        t2 = (la + mu) ./ (mu * wTw - rho_screen_avg).^2;
        den = 1 + (la + mu) .* (1 ./ (mu * wTw - rho_screen_avg)) .* wTw;
        t2 = t2 .* (1 ./ den);
    
        rhs = zeros (n^2 * 2, 1);
        rhs(1:2:end) = gradx_hat(:);
        rhs(2:2:end) = grady_hat(:);
        
        u_hat = t1 .* rhs - t2 .* (W * W.'* rhs);
        ux_hat = reshape(u_hat(1:2:end), [n, n]);
        uy_hat = reshape(u_hat(2:2:end), [n, n]);

        ux = ifftn(ux_hat,[], 'symmetric');
        uy = ifftn(uy_hat,[], 'symmetric');
        ux = real(ux);
        uy = real(uy);        
        op = [ux(:); uy(:)];
    else
        gradx_hat = fftn(fx);
        grady_hat = fftn(fy);
        gradz_hat = fftn(fz);

        wx = n_misc.i_omega_x;
        wy = n_misc.i_omega_y;
        wz = n_misc.i_omega_z;
        
        W = sparse(n^3 * 3, n^3 * 1);
        odd_entries = spdiags(wx(:), 0, n^3, n^3);
        even_entries = spdiags(wy(:), 0, n^3, n^3);
        oddeven_entries = spdiags(wz(:), 0, n^3, n^3);
        W(1:3:end,:) = odd_entries;
        W(2:3:end,:) = even_entries;
        W(3:3:end,:) = oddeven_entries;
        
        wx = wx(:); wy = wy(:); wz = wz(:);
        wTw = wx.^2 + wy.^2 + wz.^2;
        temp_mat = [wTw.'; wTw.'; wTw.'];
        wTw = reshape (temp_mat, 1, [])';
        t1 = mu * wTw - rho_screen_avg;
        t1 = 1 ./ t1;
        
        t2 = (la + mu) ./ (mu * wTw - rho_screen_avg).^2;
        den = 1 + (la + mu) .* (1 ./ (mu * wTw - rho_screen_avg)) .* wTw;
        t2 = t2 .* (1 ./ den);
    
        rhs = zeros (n^3 * 3, 1);
        rhs(1:3:end) = gradx_hat(:);
        rhs(2:3:end) = grady_hat(:);
        rhs(3:3:end) = gradz_hat(:);
        
        u_hat = t1 .* rhs - t2 .* (W * W.'* rhs);
        ux_hat = reshape(u_hat(1:3:end), [n, n, n]);
        uy_hat = reshape(u_hat(2:3:end), [n, n, n]);
        uz_hat = reshape(u_hat(3:3:end), [n, n, n]);
        
        ux = ifftn(ux_hat,[], 'symmetric');
        uy = ifftn(uy_hat,[], 'symmetric');
        uz = ifftn(uz_hat,[], 'symmetric');
        
        ux = real(ux);
        uy = real(uy);
        uz = real(uz);
        
        op = [ux(:); uy(:); uz(:)];
    end
    
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
    
    wm = wm .* (1 - c0); gm = gm .* (1 - c0); csf = csf .* (1 - c0);
    
    bg = 1 - (csf + wm + gm + c0);
    
    sz = size(c0);
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

    % Define Lam`e coefficients for the different brain tissue types
    % Use standard tissue constants
    mu_bg = 15000 / (2 * (1 + 0.48));
    lam_bg = 2 * mu_bg * 0.48 / (1 - 2 * 0.48);
    
    mu_csf = 100 / (2 * (1 + 0.1));
    lam_csf = 2 * mu_csf * 0.1 / (1 - 2 * 0.1);
    
    mu_ht = 2100 / (2 * (1 + 0.4));
    lam_ht = 2 * mu_ht * 0.4 / (1 - 2 * 0.4);
    
    mu_tu = 4000 / (2 * (1 + 0.4));
    lam_tu = 2 * mu_tu * 0.4 / (1 - 2 * 0.4);
    
    mu = mu_bg .* bg + mu_ht .* (gm + wm) + mu_tu .* c0 + mu_csf .* csf;
    lam = lam_bg .* bg + lam_ht .* (gm + wm) + lam_tu .* c0 + lam_csf .* csf;
    
    rho_screen = 1e3 .* (c0 < 0.05);
    rho_screen = rho_screen + 1e6 .* bg;
    
    
    % General domain info
    n_misc = struct('n', n, 'dim', dim, 'i_omega_x', i_omega_x, 'i_omega_y', i_omega_y, ...
                            'i_omega_z', i_omega_z, 'wm', wm, 'gm', gm, ...
                            'csf', csf, 'c0', c0, 'bg', bg, 'mu', mu, ...
                            'lam', lam, 'rho_screen', rho_screen);
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

function op = computeDiv(x, y, z, n_misc)
    [gx, ~, ~] = computeGradient(x, n_misc);
    [~, gy, ~] = computeGradient(y, n_misc);
    [~, ~, gz] = computeGradient(z, n_misc);
    
    op = gx + gy + gz;
end

function [fx, fy, fz] = computeForce(c, n_misc)
    c = imgaussfilt(c, 5);
    [fx, fy, fz] = computeGradient(c, n_misc);
    beta = 1E4;
    fx = beta .* fx .* tanh(c);
    fy = beta .* fy .* tanh(c);
    fz = beta .* fz .* tanh(c);
end
    




