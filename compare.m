% test data
load('head_idd.mat');
idd_i = z2d(4981,:)'; % 214 454 1587 1724 4981 6342
depth = z';
num_bp = 3;

z_max = max(depth);
[v,i] = maxk(abs(diff(medfilt1(idd_i,3))),num_bp);
%para0 = [zr,0.07*zr,1e-3,zv*zr*1e-2];% prior
x0 = zeros(4*num_bp,1);
lb = repmat([0,0,-10,0],1,num_bp)';
ub = repmat([1.2*z_max 10, 10, 10],1,num_bp)';
zv = v;
zr = depth(i);
x0(1:4:end) = zr;
x0(2:4:end) = 0.07*zr;
x0(3:4:end) = 1e-3;
x0(4:4:end) = zv.*zr.*1e-2;
%% fit
T = 200;
lr = 1e-2;
[x1,loss1] = adam(depth,x0,idd_i,lb,ub,lr,T);
idd_o1 = bf_mex(depth,x1,'idd');
[x2,loss2] = adai(depth,x0,idd_i,lb,ub,lr,T);
idd_o2 = bf_mex(depth,x2,'idd');
%% show results
figure
plot(depth,idd_i);hold on
plot(depth,idd_o1,'r')
plot(depth,idd_o2,'b-.')
figure
plot(loss1);hold on
plot(loss2);hold on
%%
function [theta_best,loss] = adam(depth,para,idd_i,lb,ub,lr,T)
%     T = 2000;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    loss = zeros(T,1);
    m_tm1 = 0;
    v_tm1 = 0;
    theta_tm1 = para;
    
    theta_best = para;
    loss_best = 1e9;
    loss(1) = norm((bf_mex(depth,theta_tm1,'idd') - idd_i),'fro');
    for t = 2:T
        % get gradient = jacobian*error
        g_t = 2*bf_mex(depth,theta_tm1,'jacobian')'*(bf_mex(depth,theta_tm1,'idd') - idd_i);
        % Update biased first moment estimate
        m_t = beta1*m_tm1 + (1-beta1)*g_t;
        % Update biased second raw moment estimate
        v_t = beta2*v_tm1 + (1-beta2)*g_t.^2;
        % Compute bias-corrected first moment estimate
        m_t_hat = m_t / (1-beta1^(t-1));
        % Compute bias-corrected second raw moment estimate
        v_t_hat = v_t / (1-beta2^(t-1));
        % Update parameters
        theta_t = theta_tm1 - lr*m_t_hat./(sqrt(v_t_hat)+epsilon);
        
        % constrain
        theta_t(theta_t < lb) = lb(theta_t < lb);
        theta_t(theta_t > ub) = ub(theta_t > ub);
        
        theta_tm1 = theta_t;
        m_tm1 = m_t;
        v_tm1 = v_t;
            
        idd_pred = bf_mex(depth,theta_t,'idd');
        loss(t) = norm((idd_pred - idd_i),'fro');
        
        if loss(t) < loss_best
           loss_best = loss(t);
           theta_best = theta_t;
%            plot(idd_i);hold on
%            plot(idd_pred,'-.')
%            pause(0.01)
%            clf
        end
%         if (abs(loss(t) - loss(t-1)) < 1e-6)
%             break;
%         end
    end
    
end
function [theta_best,loss] = adai(depth,para,idd_i,lb,ub,lr,T)
    % adam inertia
%     T = 2000;
    beta0 = 0.1;
    beta1_cum_prod = 1;
    beta2 = 0.99;
    epsilon = 1e-3;
    loss = zeros(T,1);
    m_tm1 = 0;
    v_tm1 = 0;
    theta_tm1 = para;
    v_t_mean = 0;
    
    theta_best = para;
    loss_best = 1e9;
    loss(1) = norm((bf_mex(depth,theta_tm1,'idd') - idd_i),'fro');
    for t = 2:T
        % get gradient = jacobian*error
        g_t = 2*bf_mex(depth,theta_tm1,'jacobian')'*(bf_mex(depth,theta_tm1,'idd') - idd_i);
        % Update biased second raw moment estimate
        v_t = beta2*v_tm1 + (1-beta2)*g_t.^2;
        % Compute bias-corrected second raw moment estimate
        v_t_hat = v_t / (1-beta2^(t-1));
        v_t_mean = mean(v_t_hat);
        beta1t = max(min(1-(v_t_hat./v_t_mean).*beta0, 1-epsilon),0);
        % Update biased first moment estimate
        m_t = beta1t.*m_tm1 + (1-beta1t).*g_t;
        beta1_cum_prod = beta1_cum_prod.*beta1t;
        % Compute bias-corrected first moment estimate
        m_t_hat = m_t ./ (1-beta1_cum_prod);
        % Update parameters
        theta_t = theta_tm1 - lr*m_t_hat;
        % constrain
        theta_t(theta_t < lb) = lb(theta_t < lb);
        theta_t(theta_t > ub) = ub(theta_t > ub);
        
        theta_tm1 = theta_t;
        m_tm1 = m_t;
        v_tm1 = v_t;
            
        idd_pred = bf_mex(depth,theta_t,'idd');
        loss(t) = norm((idd_pred - idd_i),'fro');
        
        if loss(t) < loss_best
           loss_best = loss(t);
           theta_best = theta_t;
%            plot(idd_i);hold on
%            plot(idd_pred,'-.')
%            pause(0.01)
%            clf
        end
%         if (abs(loss(t) - loss(t-1)) < 1e-6)
%             break;
%         end
    end
    
end