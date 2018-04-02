function [a,b] = SeparatingHyperplane(s,p)
[dim N] = size(p);
for i = 1:N
    A_11(i,1:dim) = -s(i)*p(:,i);
    A_11(i,dim+1) = -s(i);
end
A_12 = -eye(N);
A_21 = zeros(N,dim+1);
A_22 = -eye(N);
A = [A_11 A_12;A_21 A_22];
B = [-ones(N,1);zeros(N,1)];
cvx_begin
c = [zeros(dim,1);zeros(1,1);ones(N,1)];
variables a(dim,1) b(1,1) u(N,1)
minimize (c'*[a;b;u])
subject to
A*[a;b;u]<=B;
cvx_end
end