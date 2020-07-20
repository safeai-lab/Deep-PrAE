%% Comparison of Cross Entropy (CE) and Adaptive Multilevel Splitting (AMS)

clear; 
rng(123)

%% Parameters
n = 50000;
B = 5;
m = 2;

s1 = 1; % rare-event set sensitivity: larger = higher sensitivity
s2 = 100;

% Function defining rare event set boundary
R_1 = @(x) (sqrt(sum((x - [5,0]).^2,2)) - 1)/s1;
R_2 = @(x) (sqrt(sum((x - [0,5]).^2,2)) - 0.5)/s2;
R = @(x) -min(R_1(x),R_2(x)); % > 0 is in rare event set

gamma_start = 4; % end at 0

uniform_samples = B*rand(n,m);

sigma_cov = 0.5:0.05:1;
n_sig = length(sigma_cov);
IS_vec = zeros(n_sig,1);
AMS_vec = zeros(n_sig,1);
CE_vec = zeros(n_sig,1);

%% Implementation of CE and AMS for the given rare event set
for i_sig=1:n_sig
    
    sigma_now = sigma_cov(i_sig);

    Z_u = (R(uniform_samples)>0).*mvnpdf(uniform_samples,zeros(1,2),eye(2)*sigma_now)./mvncdf([0 0],[B B],zeros(1,2),eye(2)*sigma_now).*B^m;
    ciwidth_Z_u = 1.96*std(Z_u)/sqrt(n);
    mean_IS = mean(Z_u);
    IS_vec(i_sig) = mean_IS;
    
    % CE
    n_step = 1;
    max_step = 100;
    N_explore = 1000;
    a_p = [0,0];

    a_record = zeros(1,2);
    gamma_now = 1;
    while (n_step < max_step) && (gamma_now ~= 0)

        temp_saples = mvnrnd(a_p,eye(2)*sigma_now,N_explore*10);
        ok_samples = temp_saples(prod(temp_saples>0 & temp_saples<B,2)>0,:);
        ce_samples = ok_samples(1:N_explore,:);



        gamma_now = min(0,quantile(R(ce_samples),0.9));

        in_rare_ind = R(ce_samples)>gamma_now;

        % CE

        c_p = in_rare_ind.*mvnpdf(ce_samples,[0,0],eye(2)*sigma_now)./mvnpdf(ce_samples,a_p,eye(2)*sigma_now);

        a_p = sum(repmat(c_p,1,2).*ce_samples)/sum(c_p);

        a_record(n_step,:) = a_p;
        n_step = n_step+1;
    end

    temp_saples = mvnrnd(a_p,eye(2)*sigma_now,n*10);
    ok_samples = temp_saples(prod(temp_saples>0 & temp_saples<B,2)>0,:);
    ce_samples = ok_samples(1:n,:);
    in_rare_ind = R(ce_samples)>gamma_now;
    c_p = in_rare_ind.*mvnpdf(ce_samples,[0,0],eye(2)*sigma_now)./mvnpdf(ce_samples,a_p,eye(2)*sigma_now);

    ce_mean = mean(c_p);
    ce_std = 1.96*std(c_p)/sqrt(n);
    
    CE_vec(i_sig) = ce_mean;
    
    % AMS
    
    rng(321)
    n_step = 1;
    max_step = 100;

    N_explore = 1000;
    a_p = [0,0];

    p_record = zeros(1,2);

    temp_saples = mvnrnd([0,0],eye(2)*sigma_now,N_explore*10);
    ok_samples = temp_saples(prod(temp_saples>0 & temp_saples<B,2)>0,:);
    split_samples = ok_samples(1:N_explore,:);
    gamma_now = min(0,quantile(R(split_samples),0.9));
    num_good = sum(R(split_samples)>gamma_now);
    good_samples = split_samples(R(split_samples)>gamma_now,:);
    pick_ind = randi(num_good,N_explore-num_good,1);
    split_samples(R(split_samples)<gamma_now,:) = good_samples(pick_ind,:);
    ams_mean = 0.1;
    delta = 0.05;
    while (n_step < max_step) && (gamma_now~=0)

        for i_mh = 1:1000
            proposal_samples = split_samples+((rand(N_explore,2)-0.5)*2)*delta;
            A_samples = min((mvnpdf(proposal_samples,zeros(1,2),eye(2)*sigma_now).*(R(proposal_samples)>gamma_now).*(prod(proposal_samples>0 & proposal_samples<B,2)>0))./(mvnpdf(split_samples,zeros(1,2),eye(2)*sigma_now).*(R(split_samples)>gamma_now).*(prod(split_samples>0 & split_samples<B,2)>0)),1);
            u_samples = rand(N_explore,1);    
            accept_ind = u_samples<A_samples;
            split_samples(accept_ind,:) = proposal_samples(accept_ind,:);
        end
        
        gamma_now = min(0,quantile(R(split_samples),0.9));
        num_good = sum(R(split_samples)>gamma_now);

        good_samples = split_samples(R(split_samples)>gamma_now,:);
        pick_ind = randi(num_good,N_explore-num_good,1);
        split_samples(R(split_samples)<gamma_now,:) = good_samples(pick_ind,:);

        n_step = n_step+1;
        ams_mean = ams_mean*(num_good/N_explore);
    end
    AMS_vec(i_sig) = ams_mean;
end

%% Plot the results

figure(23)
plot(sigma_cov,IS_vec,'LineWidth',2)
hold on
plot(sigma_cov,CE_vec,'--','LineWidth',2)
plot(sigma_cov,AMS_vec,':','LineWidth',2)
legend('Truth','CE','AMS')
set(gca, 'YScale', 'log')
xlabel('\sigma^2')
ylabel('Probability Estimate')